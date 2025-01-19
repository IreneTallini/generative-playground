"This code implements a simple score matching model for the moons dataset"
from matplotlib import pyplot as plt
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch import vmap
from torch.func import grad

from datasets import (
    ContrastiveMoonDataset,
    MoonDatasetWithNoise,
)
from modules import MoonClassifier, DenoiserModel
from utils import plot_classifier_decision_boundary

BATCH_SIZE = 64
SIGMA_MIN = 0.0001
SIGMA_MAX = 2.0
RHO = 7
N_SAMPLES_TRAIN = 10000
N_SAMPLES_VAL = 500
NUM_EPOCHS_TRAIN = 100
NUM_TIMESTEPS = 10000
DEVICE = "cuda:0"
LR = 1e-5
CLASSIFIER_CKPT = "classifier.pt"
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
    dataset = MoonDatasetWithNoise(num_samples=N_SAMPLES_TRAIN)
    
    val_dataset = MoonDatasetWithNoise(num_samples=N_SAMPLES_VAL)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Implement checkpointing
    ckpt_path = Path(SCORE_CKPT)
    maybe_load_checkpoint(model, ckpt_path)
    
    # Initialize plot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    train_losses = []
    val_losses = []

    for _ in tqdm(range(NUM_EPOCHS_TRAIN)):
        for x_noisy, x, t in data_loader:
            x_noisy, x, t = x_noisy.to(DEVICE), x.to(DEVICE), t.to(DEVICE)
            denoised_x = model(x_noisy=x_noisy, t=t)
            loss = torch.mean((x - denoised_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update train loss plot
            train_losses.append(loss.item())
            ax1.clear()
            ax1.plot(train_losses, label="Train Loss")
            ax1.legend()
        
        # Validation
        val_loss = 0
        with torch.no_grad():
            for x_noisy_val, x_val, t_val in val_loader:
                x_noisy_val, x_val, t_val = x_noisy_val.to(DEVICE), x_val.to(DEVICE), t_val.to(DEVICE)
                denoised_x_val = model(x_noisy=x_noisy_val, t=t_val)
                val_loss += torch.mean((x_val - denoised_x_val) ** 2).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update validation loss plot
        ax2.clear()
        ax2.plot(val_losses, label="Val Loss")
        ax2.legend()
        plt.savefig("train_score_model.png")
    
    torch.save(model.state_dict(), ckpt_path)
    return model


def train_classifier():
    """Trains the MoonClassifier on the ContrastiveMoonDataset and saves a checkpoint."""
    model = MoonClassifier()
    dataset = ContrastiveMoonDataset(num_samples=N_SAMPLES_TRAIN)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataset = ContrastiveMoonDataset(num_samples=N_SAMPLES_VAL)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt_path = Path(CLASSIFIER_CKPT)
    maybe_load_checkpoint(model, ckpt_path)
    
    # Initialize plot
    _, (ax1, ax2) = plt.subplots(2, 1)
    train_losses = []
    val_losses = []
    val_accuracies = []

    for _ in tqdm(range(NUM_EPOCHS_TRAIN)):
        for x, label in data_loader:
            # transform in one hot encoding
            label = torch.nn.functional.one_hot(
                label, num_classes=2
            ).to(torch.float32)
            logits = model(x)
            # cross entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update train loss plot
            train_losses.append(loss.item())
            ax1.clear()
            ax1.plot(train_losses, label="Train Loss")
            ax1.legend()
        
        # Validation
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, label_val in val_loader:
                label_val = torch.nn.functional.one_hot(
                    label_val, num_classes=2
                ).to(torch.float32)
                logits_val = model(x_val)
                val_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits_val, label_val).item()
                preds = torch.argmax(logits_val, dim=1)
                correct += (preds == torch.argmax(label_val, dim=1)).sum().item()
                total += label_val.size(0)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        # Update validation loss and accuracy plot
        ax2.clear()
        ax2.plot(val_losses, label="Val Loss")
        ax2.plot(val_accuracies, label="Val Accuracy")
        ax2.legend()
        plt.savefig("train_classifier_metrics.png")
    
    torch.save(model.state_dict(), ckpt_path)
    return model


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
    target_label = 0
    ckpt_path = Path(CLASSIFIER_CKPT)
    classifier = MoonClassifier().to(device)
    classifier.load_state_dict(torch.load(ckpt_path))
    classifier.eval()
    
    def f(x):
        softmax = torch.nn.functional.softmax(classifier(x), dim=0)
        return softmax[target_label]
    
    grad_f = grad(f)
    batched_grad_f = vmap(grad_f)

    alpha = 0.001
    x = np.random.uniform(size=(N_SAMPLES_VAL, 2), low=-5, high=5)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    for _ in tqdm(range(NUM_TIMESTEPS)):
        gradient = batched_grad_f(x)
        x = x - alpha * gradient
    return x


if __name__ == "__main__":
    print("Training score model")
    score_model = train_score_model()
    print("Training classifier")
    classifier = train_classifier()  
    plot_classifier_decision_boundary(classifier, -5, 5, -5, 5, "classifier_decision_boundary.png")
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
