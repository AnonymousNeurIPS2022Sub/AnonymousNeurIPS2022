import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_goal(goal_gaussian, samples, save=False, file=None, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5):
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

    z2 = goal_gaussian.log_prob(inp.double()).exp()
    z2 = z2.view(y.shape[0], y.shape[1])
    fig, ax = plt.subplots(1, 1)
    ax.contourf(xs, ys, z2, levels=20)

    for set in samples:
        ax.scatter(set[:, 0], set[:, 1])

    if not save:
        plt.show()
    else:
        plt.savefig(f"{file}/goal.png")