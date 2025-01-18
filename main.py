"This code implements a simple score matching model for the moons dataset"
from matplotlib import pyplot as plt
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
NUM_EPOCHS_TRAIN = 10
NUM_TIMESTEPS = 10000
DEVICE = "cpu"
LR = 1e-5
CLASSIFIER_CKPT = "classifier_contrastive.pt"
SCORE_CKPT = "score_model.pt"
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


def maybe_load_checkpoint(model, ckpt_path):
    """Loads model weights if checkpoint file exists."""
    if ckpt_path.exists():
        print(f"Loading model from checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))


def train_score_model():
    """Trains the DenoiserModel on the MoonDataset and saves a checkpoint."""
    model = DenoiserModel(sigma_schedule=SIGMA_SCHEDULE, device=DEVICE)
    dataset = MoonDataset(num_samples=N_SAMPLES_TRAIN)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Implement checkpointing
    ckpt_path = Path(SCORE_CKPT)
    maybe_load_checkpoint(model, ckpt_path)
    for _ in tqdm(range(NUM_EPOCHS_TRAIN)):
        for x_noisy, x, t in data_loader:
            x_noisy, x, t = x_noisy.to(DEVICE), x.to(DEVICE), t.to(DEVICE)
            denoised_x = model(x_noisy=x_noisy, t=t)
            loss = torch.mean((x - denoised_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), ckpt_path)


def train_classifier():
    """Trains the MoonClassifier on the ContrastiveMoonDataset and saves a checkpoint."""
    model = MoonClassifier()
    data_provider = ContrastiveMoonDataset(num_samples=N_SAMPLES_TRAIN)
    data_loader = torch.utils.data.DataLoader(
        data_provider, batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt_path = Path(CLASSIFIER_CKPT)
    maybe_load_checkpoint(model, ckpt_path)
    for _ in tqdm(range(NUM_EPOCHS_TRAIN)):
        for x, label in data_loader:
            # transform in one hot encoding
            label = torch.nn.functional.one_hot(
                label, num_classes=2
            ).to(torch.float32)
            pred = model(x)
            # cross entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), ckpt_path)


def sample_with_score():
    """Samples from the trained score model and returns the generated data."""
    device = "cpu"
    
    # Load score model
    score_ckpt_path = Path(SCORE_CKPT)
    model = DenoiserModel(sigma_schedule=SIGMA_SCHEDULE, device=device)
    maybe_load_checkpoint(model, score_ckpt_path)
    
    x_guided = model.sample(n=N_SAMPLES_VAL)
    return x_guided


def sample_with_classifier_logit_search():
    """Performs logit-based gradient search using the classifier."""
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
    x = np.random.uniform(size=(N_SAMPLES_VAL, 2), low=-5, high=5)
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    for _ in tqdm(range(10000)):
        grad = batched_jacobian(x).squeeze(1)
        x = x - alpha * grad
    return x


if __name__ == "__main__":
    print("Training score model")
    train_score_model()
    print("Training classifier")
    train_classifier()
    print("Sampling from score model")
    x_score = sample_with_score().detach().numpy()
    print("Sampling from classifier")
    x_class = sample_with_classifier_logit_search().detach().numpy()
    
    # scatter plot of the two methods
    plt.figure()
    plt.scatter(x_class[:, 0], x_class[:, 1], c="r", label="Classifier")
    plt.scatter(x_score[:, 0], x_score[:, 1], c="b", label="Score")
    plt.legend()
    plt.savefig("comparison.png")
