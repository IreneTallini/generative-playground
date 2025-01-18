import torch
import matplotlib.pyplot as plt
def grid_plot(classifier, x):
    n_points_grid = 100
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-5, 5, n_points_grid), torch.linspace(-5, 5, n_points_grid)
    )
    grid_points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1)
    logits_grid = classifier(grid_points).detach().numpy()
    plt.figure()
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c=logits_grid[:, 0])
    plt.colorbar()
    plt.scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], c="r", s=3)
    plt.savefig("sample_logit_search.png")