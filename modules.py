import torch
from torch import nn
from typing import List
from tqdm import tqdm


class MoonClassifier(nn.Module):
    """MLP classifier for two-class outputs."""
    def __init__(self, input_size=2, hidden_sizes=[64, 128, 256], output_size=2, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Linear(
                    input_size, hidden_sizes[0], device=device
                ),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device),
                    nn.ReLU(),
                )
                for i in range(len(hidden_sizes) - 1)
            ],
            nn.Linear(hidden_sizes[-1], output_size, device=device),
        )

    def forward(self, x):
        return self.net(x)


class DenoiserModel(nn.Module):
    """Denoiser model implementing score matching as in https://arxiv.org/abs/2206.00364."""
    def __init__(
        self,
        input_size=2,
        output_size=2,
        time_embedding_size=2,
        hidden_sizes: List[int] = [64, 128, 256],
        sigma_schedule=None,
        batch_size=64,
        device="cpu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Linear(
                    input_size + time_embedding_size, hidden_sizes[0], device=device
                ),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device),
                    nn.ReLU(),
                )
                for i in range(len(hidden_sizes) - 1)
            ],
            nn.Linear(hidden_sizes[-1], output_size, device=device),
        )
        self.time_embedding_net = nn.Sequential(
            nn.Linear(1, hidden_sizes[0], device=device),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device),
                    nn.ReLU(),
                )
                for i in range(len(hidden_sizes) - 1)
            ],
            nn.Linear(hidden_sizes[-1], time_embedding_size, device=device),
        )
        self.sigma_schedule = sigma_schedule
        self.num_steps = len(sigma_schedule)
        self.batch_size = batch_size
        self.device = device

    def forward(self, t, x_noisy):
        """Computes the denoised sample given a noisy input at time t."""
        t_embeddings = self.time_embedding_net(t)
        x_noisy_time = torch.cat([x_noisy, t_embeddings], dim=1)
        x_denoised = self.net(x_noisy_time)
        return x_denoised

    def score(self, x, t):
        """Computes the score function used for sampling."""
        t_embeddings = self.time_embedding_net(t)
        x_t = torch.cat([x, t_embeddings], dim=1)
        x_denoised = self.net(x_t)
        score = (
            -(x_denoised - x) / t
        )  # The minus is because we are integrating backward
        return score

    @torch.no_grad()
    def sample(self, n=100):
        """Generates samples by iterating through the sigma schedule."""
        x = self.sigma_schedule[0] * torch.randn((n, 2), device=self.device)
        for i in tqdm(range(self.num_steps - 1)):
            t_more_noisy = torch.tensor([self.sigma_schedule[i]], device=self.device)
            t_less_noisy = torch.tensor(
                [self.sigma_schedule[i + 1]], device=self.device
            )
            # compute the score
            score = self.score(x, t_more_noisy.repeat(n, 1))
            x = x + score * (t_less_noisy - t_more_noisy)
            # add stochasticity to sampling
            if i != self.num_steps - 1:
               x = x + torch.sqrt(t_more_noisy ** 2 - t_less_noisy ** 2) * torch.randn_like(x)
        return x
