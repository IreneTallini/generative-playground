import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_classifier_decision_boundary(classifier, x_min, x_max, y_min, y_max, filename):
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.tensor(grid, dtype=torch.float32)
    grid_logits = classifier(grid).detach().numpy()
    grid_labels = np.argmax(grid_logits, axis=1)
    plt.figure()
    plt.scatter(grid[:, 0], grid[:, 1], c=grid_labels, alpha=0.5)
    plt.colorbar()
    plt.savefig(filename)
    
def plot_classifier_dataset(dataset):
    plt.figure()
    x, label = dataset[:]
    plt.scatter(x[:, 0], x[:, 1], c=label)
    plt.savefig("moon_dataset.png")