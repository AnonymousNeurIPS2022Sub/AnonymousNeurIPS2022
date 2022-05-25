import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_potential(potential, samples, save=False, file=None, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5):
    """

    :param potential:
    :param samples: List of sample sets
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :return:
    """

    xs = np.arange(xmin, xmax, .1)
    ys = np.arange(ymin, ymax, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor([x, y]).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])
    fig, ax = plt.subplots(1, 1)
    ax.contourf(xs, ys, z, levels=10)
    for set in samples:
        ax.scatter(set[:, 0], set[:, 1])

    if not save or file is None:
        plt.show()
    else:
        plt.savefig(f"{file}/landscape.png")


def plot_quiver(potential, save=False, file=None, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5):
    """

    :param potential:
    :param samples: List of sample sets
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :return:
    """

    xs = np.arange(xmin, xmax, .1)
    ys = np.arange(ymin, ymax, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor([x, y]).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])
    drift = potential.drift(inp)

    norm = torch.linalg.norm(drift, dim=1)
    u = (drift[:, 0] * 0.1 / norm).view(y.shape[0], y.shape[1])
    v = (drift[:, 1] * 0.1 / norm).view(y.shape[0], y.shape[1])

    fig, ax = plt.subplots(1, 1)
    ax.quiver(x, y, u, v, units='xy', scale=1, color='gray')
    ax.contour(x, y, z, 7, cmap='jet')

    if not save:
        plt.show()
    else:
        plt.savefig(f"{file}/prior_drift.png")
