from sklearn import datasets
import numpy as np
import torch
from torch.utils.data import Dataset   

class MoonDatasetWithNoise(Dataset):
    """Moon dataset with random noise injection."""
    def __init__(self, num_samples):
        self.mean = torch.tensor([0.5, 0.25])
        self.num_samples = num_samples
        data, labels = datasets.make_moons(n_samples=num_samples)
        data = torch.tensor(data, dtype=torch.float32)
        self.data = data - self.mean
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = torch.tensor((1.2 * np.random.rand(1) - 1.2) ** 2, dtype=torch.float32)
        x = self.data[idx]
        x_noisy = x + t * torch.randn(2, dtype=torch.float32)
        return x_noisy, x, t
    
class ContrastiveMoonDataset(Dataset):
    """Half standard moons data, half random data for contrastive classification."""
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.mean = torch.tensor([0.5, 0.25])
        data, _ = datasets.make_moons(n_samples=num_samples)
        data = torch.tensor(data, dtype=torch.float32)
        self.data = data - self.mean
        indices = np.random.choice(num_samples, num_samples // 2, replace=False)
        data_rand = torch.rand((num_samples // 2, 2)) * 10 - 5
        self.data[indices] = data_rand
        self.labels = torch.ones(num_samples, dtype=torch.long)
        self.labels[indices] = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        return x, label

class ConditionalMoonDataProvider:
    """Generates data conditioned on a noise parameter t."""
    def get_data(self, t, num_samples):
        data, labels = datasets.make_moons(n_samples=num_samples)
        data = data - np.array([0.5, 0.25])
        data_noisy = data - np.array([0.5, 0.25]) + t * np.random.randn(num_samples, 2)
        return data_noisy, data, labels