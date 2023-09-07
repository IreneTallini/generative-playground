"This code implements a simple score matching model for the moons dataset"
from functools import partial

from sklearn import datasets
import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

class MoonDataProvider():
    def __init__(self):
        self.mean = np.array([0.5, 0.25])

    def get_data(self, t=0., num_samples=64):
        data, labels = datasets.make_moons(n_samples=num_samples)
        data = data - self.mean
        noisy_data = data + t * np.random.randn(num_samples, 2)
        return noisy_data, data, labels

class MoonContrastiveDataProvider():
    def __init__(self):
        self.mean = np.array([0.5, 0.25])

    def get_data(self, t=0., num_samples=64):
        # sample a bernoulli variable
        labels = np.random.binomial(1, 0.5, size=(num_samples,)).astype(bool)
        data, _ = datasets.make_moons(n_samples=num_samples)
        data = data - self.mean
        data_rand = np.random.uniform(low=-5., high=5., size=(num_samples, 2))
        data[~labels] = data_rand[~labels]
        noisy_data = None
        return noisy_data, data, labels

class MoonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class DenoiserModel(nn.Module):
    # Implement Karras Elucidating the Design Space of Diffusion-Based Generative Models
    def __init__(self, input_size=2, output_size=2, time_embedding_size = 2, hidden_sizes: List[int] = [64, 128, 256],
                 sigma_schedule=None, batch_size=64, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(input_size + time_embedding_size, hidden_sizes[0], device=device), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], device=device), nn.ReLU())
              for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], output_size, device=device)
        )
        self.time_embedding_net = nn.Sequential(
            nn.Linear(1, hidden_sizes[0], device=device), nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], device=device), nn.ReLU())
              for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], time_embedding_size, device=device)
        )
        self.sigma_schedule = sigma_schedule
        self.num_steps = len(sigma_schedule)
        self.batch_size = batch_size
        self.device = device

    def forward(self, t, x_noisy):
        t_embeddings = self.time_embedding_net(t)
        x_noisy_time = torch.cat([x_noisy, t_embeddings], dim=1)
        x_denoised = self.net(x_noisy_time)
        return x_denoised

    def score(self, x, t):
        t_embeddings = self.time_embedding_net(t)
        x_t = torch.cat([x, t_embeddings], dim=1)
        x_denoised = self.net(x_t)
        score = - (x_denoised - x) / t # The minus is because we are integrating backward
        return score

    @torch.no_grad()
    def sample(self, n=100, target_sum=None, classifier=None, target_label=None):
        # Sample from a gaussian with std the max of self.sigma_schedule
        sigma_min = 0.0001
        sigma_max = 2.
        rho = 7
        num_timesteps = 100000
        sigma_schedule = [
            (sigma_max ** (1 / rho) + (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * i / (
                        num_timesteps - 1)) ** rho
            for i in range(num_timesteps)]
        x = self.sigma_schedule[0] * torch.randn((n, 2), device=self.device)
        for i in tqdm(range(self.num_steps - 1)):
            t_more_noisy = torch.tensor([self.sigma_schedule[i]], device=self.device)
            t_less_noisy = torch.tensor([self.sigma_schedule[i+1]], device=self.device)
            # compute the score
            # score = self.score(x, t_more_noisy.repeat(n, 1))
            x_denoised = self(x_noisy=x, t=t_more_noisy.repeat(n, 1))
            if target_sum is not None:
                def f(point):
                    point = point.unsqueeze(0)
                    return self.forward(t=t_more_noisy.unsqueeze(0), x_noisy=point)

                def batched_jacobian(points):  # [B, D] ->
                    jacobian_fn = torch.func.jacfwd(f)
                    jacobian_fn_vmap = torch.vmap(jacobian_fn)  # [B, D] -> [B, D, D]
                    jacobian = jacobian_fn_vmap(points)
                    return jacobian

                # jacobian = batched_jacobian(x).squeeze(1)
                dl_df = 2 * (target_sum - x_denoised.sum(dim=1, keepdims=True))
                # df_dx = - (torch.ones((x.shape[0], 1, x.shape[-1]), device=self.device) @ jacobian).squeeze(1)
                df_dx = - torch.ones((x.shape[0], 1, x.shape[-1]), device=self.device).squeeze(1)
                score = score + dl_df * df_dx
                print(f"{x_denoised.sum(dim=1).mean()=}")
            if classifier is not None:
                def f(denoised_point):
                    pred = classifier(denoised_point)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_label)
                    return loss

                def batched_autograd(points):  # [B, D] ->
                    jacobian_fn = torch.func.jacfwd(f)
                    jacobian_fn_vmap = torch.vmap(jacobian_fn)  # [B, D] -> [B, D, D]
                    jacobian = jacobian_fn_vmap(points)
                    return jacobian

                # gradient = batched_autograd(x_denoised).squeeze(1)
                gradient = batched_autograd(x).squeeze(1)
                score = gradient
                # score = score + gradient
            # update x_noisy though heun discretization
            x = x + score * (t_less_noisy - t_more_noisy)
            #if i != self.num_steps - 1:
            #    x = x + torch.sqrt(t_more_noisy ** 2 - t_less_noisy ** 2) * torch.randn_like(x)
        return x

def train_score_model():
    device = "cuda:0"
    batch_size = 64
    sigma_min = 0.0001
    sigma_max = 2.
    rho = 7
    num_timesteps = 10000
    print(f"num_timesteps={num_timesteps}")
    sigma_schedule = [
        (sigma_max ** (1 / rho) + (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * i / (num_timesteps - 1)) ** rho
        for i in range(num_timesteps)]
    model = DenoiserModel(sigma_schedule=sigma_schedule, device=device)
    data_provider = MoonDataProvider()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Implement checkpointing
    ckpt_path = Path("model_10000ts.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
    else:
        for i in range(10000):
            t = (1.2 * np.random.rand(batch_size, 1) - 1.2) ** 2
            x_noisy, x = data_provider.get_data(t=t)
            x_noisy_t = torch.tensor(x_noisy, dtype=torch.float32, device=device)
            x_t = torch.tensor(x, dtype=torch.float32, device=device)
            t_t = torch.tensor(t, dtype=torch.float32, device=device)
            denoised_x = model(x_noisy=x_noisy_t, t=t_t)
            # compute the l2 loss between x_noisy and x_denoised
            loss = torch.mean((x_t - denoised_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss)
        torch.save(model.state_dict(), ckpt_path)

    # sample from the model
    n = 500
    offset = 0.
    # x = model.sample(n=n).cpu().detach().numpy()
    x_guided = model.sample(n=n, target_sum=offset).cpu().detach().numpy()
    _, x_gt = data_provider.get_data(t=np.array([[0.]] * n), num_samples=n)
    # plt.scatter(x[:, 0], x[:, 1], c='r', s=1)
    # plt.scatter(x_gt[:, 0], x_gt[:, 1], c='b', s=1)
    # plt.savefig('sample_uncond.png')
    # plt.clf()
    plt.scatter(x_gt[:, 0], x_gt[:, 1], c='b', s=1)
    plt.scatter(x_guided[:, 0], x_guided[:, 1], c='g', s=5)
    xx = np.linspace(-1.2, 1.2, 100)
    plt.plot(xx, - xx + offset, c='r')
    plt.savefig('sample_cond.png')

def train_classifier():
    model = MoonClassifier()
    data_provider = MoonContrastiveDataProvider()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_timesteps = 50000
    ckpt_path = Path("classifier_contrastive.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
    else:
        for i in tqdm(range(num_timesteps)):
            _, x, label = data_provider.get_data()
            x_t = torch.tensor(x, dtype=torch.float32)
            label_t = torch.tensor(label, dtype=torch.float32)
            # transform in one hot encoding
            label_t = torch.nn.functional.one_hot(label_t.to(torch.int64), num_classes=2).to(torch.float32)
            pred = model(x_t)
            # cross entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                with torch.no_grad():
                    _, x, label = data_provider.get_data()
                    x_t = torch.tensor(x, dtype=torch.float32)
                    label_t = torch.tensor(label, dtype=torch.float32)
                    # transform in one hot encoding
                    label_t = torch.nn.functional.one_hot(label_t.to(torch.int64), num_classes=2).to(torch.float32)
                    pred = model(x_t)
                    # cross entropy loss
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label_t)
                    acc = torch.sum(pred.argmax(1) == label_t.argmax(1)) / label_t.shape[0]
                    print(f"{loss=}")
                    print(f"{acc=}")
    torch.save(model.state_dict(), ckpt_path)

def cond_sample():
    target_label = 1
    n = 500
    device = "cpu"
    batch_size = 64
    sigma_min = 0.0001
    sigma_max = 2.
    rho = 7
    num_timesteps = 10000
    print(f"num_timesteps={num_timesteps}")
    sigma_schedule = [
        (sigma_max ** (1 / rho) + (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * i / (num_timesteps - 1)) ** rho
        for i in range(num_timesteps)]
    # Load model
    ckpt_path = Path(f"model_{num_timesteps}ts.pt")
    model = DenoiserModel(sigma_schedule=sigma_schedule, device=device)
    model.load_state_dict(torch.load(ckpt_path))
    # Load classifier
    ckpt_path = Path("classifier.pt")
    classifier = MoonClassifier().to(device)
    classifier.load_state_dict(torch.load(ckpt_path))
    # Sample from the model conditioned on the label
    label_t = torch.tensor(target_label, dtype=torch.float32, device=device)
    # transform in one hot encoding
    label_t = torch.nn.functional.one_hot(label_t.to(torch.int64), num_classes=2).to(torch.float32)
    x_guided = model.sample(n=n, classifier=classifier, target_label=label_t).cpu().detach().numpy()
    plt.figure()
    plt.scatter(x_guided[:, 0], x_guided[:, 1], c='g', s=5)
    data_provider = MoonDataProvider()
    _, x_gt, label = data_provider.get_data(num_samples=n)
    plt.scatter(x_gt[:, 0], x_gt[:, 1], c='b', s=1)
    plt.savefig('sample_cond_wrong.png')

def logit_search_sample():
    # Load classifier
    device = "cpu"
    ckpt_path = Path("classifier_contrastive.pt")
    classifier = MoonClassifier().to(device)
    classifier.load_state_dict(torch.load(ckpt_path))
    # perform gradient ascent on the logit
    batch_size = 64
    # sample a random uniform 2D point
    def batched_jacobian(points):  # [B, D] ->
        jacobian_fn = torch.func.jacfwd(lambda x: classifier(x)[0])
        jacobian_fn_vmap = torch.vmap(jacobian_fn)  # [B, D] -> [B, D, D]
        jacobian = jacobian_fn_vmap(points)
        return jacobian

    alpha = 0.0001
    x = np.random.uniform(size=(batch_size, 2), low=-1.2, high=1.2)
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    for i in range(1000):
        grad = batched_jacobian(x).squeeze(1)
        x = x + alpha * grad
    # plot logit landscape on a grid
    n_points_grid = 100
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1.2, 1.2, n_points_grid), torch.linspace(-1.2, 1.2, n_points_grid))
    grid_points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1)
    logits_grid = classifier(grid_points).detach().numpy()
    plt.figure()
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=logits_grid[:, 0])
    plt.colorbar()
    plt.scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], c='r', s=3)
    plt.savefig('sample_logit_search.png')

if __name__ == '__main__':
    train_classifier()
    logit_search_sample()