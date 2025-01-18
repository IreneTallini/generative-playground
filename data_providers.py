from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset   

class MoonDataset(Dataset):
    def __init__(self, num_samples):
        self.mean = np.array([0.5, 0.25])
        self.num_samples = num_samples
        data, self.labels = datasets.make_moons(n_samples=num_samples)
        self.data = data - self.mean

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, t):
        t = (1.2 * np.random.rand(1) - 1.2) ** 2
        x = self.data[idx]
        x_noisy = x + t * np.random.randn(2)
        return x_noisy, x, t
    
class ContrastiveMoonDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        data, self.labels = datasets.make_moons(n_samples=num_samples)
        self.data = data - np.array([0.5, 0.25])
        data_rand = np.random.uniform(low=-5., high=5., size=(num_samples, 2))
        self.data[~self.labels] = data_rand[~self.labels]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, t):
        t = (1.2 * np.random.rand(1) - 1.2) ** 2
        x = self.data[idx]
        x_noisy = x + t * np.random.randn(2)
        return x_noisy, x, t

class ConditionalMoonDataProvider:
    def get_data(self, t, num_samples):
        data, labels = datasets.make_moons(n_samples=num_samples)
        data = data - np.array([0.5, 0.25])
        data_noisy = data - np.array([0.5, 0.25]) + t * np.random.randn(num_samples, 2)
        return data_noisy, data, labels