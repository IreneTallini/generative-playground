"This code implements a simple score matching model for the moons dataset"
from functools import partial

from sklearn import datasets
import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MoonDataProvider():
    def __init__(self):
        self.mean = np.array([0.5, 0.25])

    def get_data(self, t, num_samples=64):
        data = datasets.make_moons(n_samples=num_samples)[0] - self.mean
        noisy_data = data + t * np.random.randn(num_samples, 2)
        return noisy_data, data


class DenoiserModel(nn.Module):
    # Implement Karras Elucidating the Design Space of Diffusion-Based Generative Models
    def __init__(self, input_size=2, output_size=2, time_embedding_size = 2, hidden_sizes: List[int] = [64, 128, 256],
                 sigma_schedule=None, batch_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(input_size + time_embedding_size, hidden_sizes[0]), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU())
              for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], output_size)
        )
        self.time_embedding_net = nn.Sequential(
            nn.Linear(1, hidden_sizes[0]), nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU())
              for i in range(len(hidden_sizes)-1)],
            nn.Linear(hidden_sizes[-1], time_embedding_size)
        )
        self.sigma_schedule = sigma_schedule
        self.num_steps = len(sigma_schedule)
        self.batch_size = batch_size

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
    def sample(self, n=100, target_sum=None):
        # Sample from a gaussian with std the max of self.sigma_schedule
        x = self.sigma_schedule[0] * torch.randn((n, 2))
        for i in range(self.num_steps - 1):
            t_more_noisy = torch.tensor([self.sigma_schedule[i]])
            t_less_noisy = torch.tensor([self.sigma_schedule[i+1]])
            # compute the score
            score = self.score(x, t_more_noisy.repeat(n, 1)) # gets overridden when target_sum is not None
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
                jacobian = batched_jacobian(x).squeeze(1)
                dl_df = 2 * (target_sum - x_denoised.sum(dim=1, keepdims=True))
                df_dx = - (torch.ones((x.shape[0], 1, x.shape[-1])) @ jacobian).squeeze(1)
                score = dl_df * df_dx
                print(f"{x_denoised.sum()=}")
            # update x_noisy though heun discretization
            x = x + score * (t_less_noisy - t_more_noisy)
            #if i != self.num_steps - 1:
            #    x = x + torch.sqrt(t_more_noisy ** 2 - t_less_noisy ** 2) * torch.randn_like(x)
        return x


if __name__ == '__main__':
    batch_size = 64
    sigma_min = 0.0001
    sigma_max = 5.
    rho = 7
    num_timesteps = 10000
    sigma_schedule = [(sigma_max**(1/rho) + (sigma_min**(1/rho) - sigma_max**(1/rho)) * i / (num_timesteps - 1))**rho
                      for i in range(num_timesteps)]
    model = DenoiserModel(sigma_schedule=sigma_schedule)
    data_provider = MoonDataProvider()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Implement checkpointing
    ckpt_path = Path("model_10000ts.pt")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
    else:
        for i in range(10000):
            t = np.random.rand(batch_size, 1)
            x_noisy, x = data_provider.get_data(t=t)
            x_noisy_t = torch.tensor(x_noisy, dtype=torch.float32)
            x_t = torch.tensor(x, dtype=torch.float32)
            t_t = torch.tensor(t, dtype=torch.float32)
            denoised_x = model(x_noisy_t, t_t)
            # compute the l2 loss between x_noisy and x_denoised
            loss = torch.mean((x_t - denoised_x) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss)
        torch.save(model.state_dict(), ckpt_path)

    # sample from the model
    n = 1000
    x = model.sample(n=n).cpu().detach().numpy()
    # x_guided = model.sample(n=n, target_sum=0.).cpu().detach().numpy()
    _, x_gt = data_provider.get_data(t=np.array([[0.]] * n), num_samples=n)
    plt.scatter(x[:, 0], x[:, 1], c='r', s=1)
    plt.scatter(x_gt[:, 0], x_gt[:, 1], c='b', s=1)
    #plt.scatter(x_guided[:, 0], x_guided[:, 1], c='g', s=5)
    plt.savefig('sample.png')
