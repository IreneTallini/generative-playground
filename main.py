"This code implements a simple score matching model for the moons dataset"
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_providers import (
    ContrastiveMoonDataset,
    MoonDataset,
)
from modules import MoonClassifier, DenoiserModel

BATCH_SIZE = 64
SIGMA_MIN = 0.0001
SIGMA_MAX = 2.0
RHO = 7
N_SAMPLES_TRAIN = 10000
N_SAMPLES_VAL = 500
NUM_TIMESTEPS = 10000
DEVICE = "cuda:0"
LR = 1e-5
CLASSIFIER_CKPT = "classifier_contrastive_10000.pt"
SCORE_CKPT = "model_10000ts.pt"
SIGMA_SCHEDULE = [
    (
        SIGMA_MAX ** (1 / RHO)
        + (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))
        * i
        / (NUM_TIMESTEPS - 1)
    )
    ** RHO
    for i in range(NUM_TIMESTEPS)
]


def train_score_model():
    """
    Trains the DenoiserModel using the MoonDataProvider and saves a checkpoint.
    """

    model = DenoiserModel(sigma_schedule=SIGMA_SCHEDULE, device=DEVICE)
    data_provider = MoonDataset(num_samples=N_SAMPLES_TRAIN, contrastive=False)
    data_loader = torch.utils.data.DataLoader(
        data_provider, batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Implement checkpointing
    ckpt_path = Path(SCORE_CKPT)
    if ckpt_path.exists():
        print("Loading model from checkpoint")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("Training model from scratch")
        for x_noisy, x, t in tqdm(data_loader):
            denoised_x = model(x_noisy=x_noisy, t=t)
            loss = torch.mean((x - denoised_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), ckpt_path)


def train_classifier():
    """
    Trains the MoonClassifier using the MoonContrastiveDataProvider and saves a checkpoint.
    """
    model = MoonClassifier()
    data_provider = ContrastiveMoonDataset(num_samples=N_SAMPLES_TRAIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt_path = Path(CLASSIFIER_CKPT)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
    else:
        for _, x, label in tqdm(data_provider):
            # transform in one hot encoding
            label = torch.nn.functional.one_hot(
                label.to(torch.int64), num_classes=2
            ).to(torch.float32)
            pred = model(x)
            # cross entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), ckpt_path)


def cond_sample():
    """
    Loads a trained model and classifier, then generates samples conditioned on a target label.
    """
    target_label = 1
    device = "cpu"
    
    # Load score model
    score_ckpt_path = Path(SCORE_CKPT)
    model = DenoiserModel(sigma_schedule=SIGMA_SCHEDULE, device=device)
    model.load_state_dict(torch.load(score_ckpt_path))
    
    # Load classifier
    classifier_ckpt_path = Path(CLASSIFIER_CKPT)
    classifier = MoonClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_ckpt_path))
    
    # Sample from the model conditioned on the label
    label_t = torch.nn.functional.one_hot(target_label.to(torch.int64), num_classes=2).to(
        torch.float32
    )
    x_guided = (
        model.sample(n=N_SAMPLES_VAL, classifier=classifier, target_label=label_t)
        .cpu()
        .detach()
        .numpy()
    )


def logit_search_sample():
    """
    Performs logit-based gradient search using the classifier and visualizes the results.
    """
    # Load classifier
    device = "cpu"
    ckpt_path = Path(CLASSIFIER_CKPT)
    classifier = MoonClassifier().to(device)
    classifier.load_state_dict(torch.load(ckpt_path))

    def batched_jacobian(points):  # [B, D] ->
        jacobian_fn = torch.func.jacfwd(lambda x: classifier(x)[0])
        jacobian_fn_vmap = torch.vmap(jacobian_fn)  # [B, D] -> [B, D, D]
        jacobian = jacobian_fn_vmap(points)
        return jacobian

    alpha = 0.001
    x = np.random.uniform(size=(BATCH_SIZE, 2), low=-5, high=5)
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    for _ in range(10000):
        grad = batched_jacobian(x).squeeze(1)
        x = x - alpha * grad


if __name__ == "__main__":
    train_classifier()
    logit_search_sample()
