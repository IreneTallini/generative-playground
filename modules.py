import torch
from torch import nn
from typing import List
from tqdm import tqdm


class MoonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


class DenoiserModel(nn.Module):
    # Implement Karras Elucidating the Design Space of Diffusion-Based Generative Models
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
        t_embeddings = self.time_embedding_net(t)
        x_noisy_time = torch.cat([x_noisy, t_embeddings], dim=1)
        x_denoised = self.net(x_noisy_time)
        return x_denoised

    def score(self, x, t):
        t_embeddings = self.time_embedding_net(t)
        x_t = torch.cat([x, t_embeddings], dim=1)
        x_denoised = self.net(x_t)
        score = (
            -(x_denoised - x) / t
        )  # The minus is because we are integrating backward
        return score

    @torch.no_grad()
    def sample(self, n=100, target_sum=None, classifier=None, target_label=None):
        assert not (classifier is not None) ^ (
            target_label is not None
        ), "classifier and target_label should be both None or not None"
        x = self.sigma_schedule[0] * torch.randn((n, 2), device=self.device)
        for i in tqdm(range(self.num_steps - 1)):
            t_more_noisy = torch.tensor([self.sigma_schedule[i]], device=self.device)
            t_less_noisy = torch.tensor(
                [self.sigma_schedule[i + 1]], device=self.device
            )
            # compute the score
            score = self.score(x, t_more_noisy.repeat(n, 1))
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
                df_dx = -(
                    torch.ones((x.shape[0], 1, x.shape[-1]), device=self.device)
                    @ jacobian
                ).squeeze(1)
                # df_dx = -torch.ones(
                #     (x.shape[0], 1, x.shape[-1]), device=self.device
                # ).squeeze(1)
                score = score + dl_df * df_dx
                print(f"{x_denoised.sum(dim=1).mean()=}")

            if classifier is not None:

                def f(denoised_point):
                    pred = classifier(denoised_point)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred, target_label
                    )
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
            # if i != self.num_steps - 1:
            #    x = x + torch.sqrt(t_more_noisy ** 2 - t_less_noisy ** 2) * torch.randn_like(x)
        return x
