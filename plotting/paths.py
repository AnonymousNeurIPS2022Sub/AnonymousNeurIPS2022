import numpy as np
import matplotlib.pyplot as plt
import torch

from plotting.BasePlotter import *


class MultiDimPathPlotter(AfterEpochPlotter):
    def __init__(self, DW, file, xmin=-1.5,
                 xmax=1.5, ymin=-1.5, ymax=1.5, interval=1):
        super().__init__(file, interval)
        self.DW = DW
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def plot(self, paths, run, us):
        xs = np.arange(self.xmin, self.xmax, .1)
        ys = np.arange(self.ymin, self.ymax, .1)
        x_, y_ = np.meshgrid(xs, ys)
        inp = torch.tensor([x_, y_]).view(2, -1).T

        # delta = 0.2
        # fac = 1.
        # fac_ = 1.

        z = self.DW.potential(inp)
        z = z.view(y_.shape[0], y_.shape[0])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(xs, ys, z, 50, cmap='binary')

        paths_in = paths[:,:,:2].view(-1, 2)
        paths_z = self.DW.potential(paths_in)
        paths_z = paths_z.view(paths.shape[0], paths.shape[1], 1)
        for r in range(0, len(paths)):
            ax.plot3D(paths[r, :, 0], paths[r, :, 1], paths_z[r, :, 0], zorder=r+5)

        # ax.plot3D(paths[:, 0, 0], paths[:, 0, 1], color='w', zorder=r+100, s=2)

        # plt.show()
        plt.savefig(f"{self.file}/{run}_3d.png")


class PathPlotter(AfterEpochPlotter):
    def __init__(self, DW, file, xmin=-1.5,
                 xmax=1.5, ymin=-1.5, ymax=1.5, interval=1):
        super().__init__(file, interval)
        self.DW = DW
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def plot(self, paths, run, us):
        xs = np.arange(self.xmin, self.xmax, .1)
        ys = np.arange(self.ymin, self.ymax, .1)
        x, y = np.meshgrid(xs, ys)
        inp = torch.tensor([x, y]).view(2, -1).T

        z = self.DW.potential(inp)
        z = z.view(y.shape[0], y.shape[1])

        fig, ax = plt.subplots(1, 1)
        ax.contourf(xs, ys, z, levels=10, zorder=0)
        for r in range(0, len(paths)):
            ax.plot(paths[r, :, 0], paths[r, :, 1], zorder=r+5)

        ax.scatter(paths[:, 0, 0], paths[:, 0, 1], color='w', zorder=r+100, s=2)

        plt.savefig(f"{self.file}/{run}.png")


class PolicyPlotter(BeforeEpochPlotter):
    def __init__(self, policy, file, T, dt, xmin=-1.5, xmax=1.5, ymin=-1.5,
                 ymax=1.5, interval=1):
        super().__init__(file, interval)
        self.policy = policy
        self.T = T
        self.dt = dt
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def plot(self, paths, run, us):

        # if run % 1 == 0:
        #     N = int((self.T / self.dt))
        #     step_folder = self.file + "/" + str(run)
        # else:
        N = 1
        step_folder = self.file

        if not os.path.exists(step_folder):
            os.makedirs(step_folder)

        xs = np.arange(self.xmin, self.xmax, .1)
        ys = np.arange(self.ymin, self.ymax, .1)
        x, y = np.meshgrid(xs, ys)

        inp = torch.tensor([x, y], dtype=torch.float).view(2, -1).T
        inp_vel = torch.zeros_like(inp, dtype=torch.float)

        inp = torch.hstack([inp, inp_vel])

        for s in np.arange(0, N, 10):
            t = s * self.dt
            drift = self.policy(inp, torch.tensor([t])).detach()
            norm = torch.linalg.norm(drift, dim=1).max()
            u = (drift[:, 0] * 0.1 / norm).view(y.shape[0], y.shape[1])
            v = (drift[:, 1] * 0.1 / norm).view(y.shape[0], y.shape[1])

            fig, ax = plt.subplots(1, 1)
            ax.quiver(x, y, u, v, units='xy', scale=1, color='gray')

            plt.title(f"step: {s}, max norm: {norm}")
            plt.savefig(f"{step_folder}/{run}_{s}_quiv.png")

            plt.close('all')


class ActionPlotter:
    def __init__(self, folder=None):
        self.folder = folder

    def plot(self, action, run):

        n = int(np.ceil(np.sqrt(action.shape[0])))
        fig, axs = plt.subplots(n, n, figsize=(3*n, 3*n), )
        step = 0
        m = action.max()
        for i in range(int(np.ceil(np.sqrt(action.shape[0])))):
            for j in range(int(np.ceil(np.sqrt(action.shape[0])))):
                u = action[step]
                norm = u.norm()
                axs[i, j].plot([0, u[0]], [0, u[1]], linewidth=norm*5)
                axs[i, j].set_xlim([-m, m])
                axs[i, j].set_ylim([-m, m])
                step += 1
            if step > action.shape[0]:
                break

        if self.folder is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(f"{self.folder}/{run}_actions.png")
