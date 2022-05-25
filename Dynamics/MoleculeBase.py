import os
from abc import abstractmethod, ABC

import numpy as np
import torch
from einops import einops

from potentials.alanine_md import AlaninePotentialMD


class MoleculeBaseDynamics(ABC):
    def __init__(self, loss_func, n_samples=10, device='cpu', bridge=False, save_file=None, bridge_samples=128):
        self.n_samples = n_samples
        self.loss_func = loss_func
        self.device = device
        self.bridge = bridge
        self.save_file = save_file
        self.bridge_samples = bridge_samples

        self.ending_positions = self._init_ending_positions()
        self.ending_positions = self.ending_positions.to(self.device)

        self.potentials = self._init_potentials()

        self.dims = self.ending_positions.shape[1] * self.ending_positions.shape[2]

        top = torch.zeros(self.dims, self.dims)
        bot = torch.eye(self.dims)  # DW.second_order(x)#torch.eye(2)

        self.G_matrix = torch.vstack([top, bot])
        self.G_matrix = einops.repeat(self.G_matrix, 'm n -> k m n', k=n_samples).to(device)


    @abstractmethod
    def _init_potentials(self):
        pass

    @abstractmethod
    def _init_ending_positions(self):
        pass

    def f(self, x, t):
        # reshaped = x.view(self.n_samples, -1, 6)
        pos = x[:, :int(x.shape[1] / 2)].view(self.n_samples, -1, 3)
        vel = x[:, int(x.shape[1] / 2):].view(self.n_samples, -1, 3)
        # vel = torch.zeros_like(vel)
        vel_np = vel.detach().cpu().numpy()

        ps = []
        for i in range(self.n_samples):
            _p, _v = self.potentials[i].drift(vel_np[i, :, :])
            ps.append(_p)

        ps = torch.tensor(np.array(ps), dtype=torch.float, device=self.device)
        dx = ps - pos
        dx_ = dx.view(self.n_samples, -1)
        dv_ = -vel.view(self.n_samples, -1)

        comb = torch.cat([dx_, dv_], dim=1)

        return comb

    def G(self, x):
        return self.G_matrix


    def q(self, x):
        return 0.
    #
    #
    # def phi(self, x):
    #     end_ = self.ending_positions.view(self.ending_positions.shape[0], -1)
    #     x_ = x[:, :int(x.shape[1] / 2)]
    #
    #     if self.loss_func == 'pairwise_dist':
    #         x_ = x_.view(x_.shape[0], -1, 3)
    #         end_ = end_.view(end_.shape[0], -1, 3)
    #         px = torch.cdist(x_, x_).unsqueeze(0)
    #         pend = torch.cdist(end_, end_).unsqueeze(1).repeat(1,
    #                                                            self.n_samples,
    #                                                            1, 1)
    #
    #         t = (px - pend) ** 2
    #         cost_distance = torch.mean(t, dim=(2, 3))
    #
    #         cost_distance_final = (cost_distance * 100).exp() * 150
    #
    #         expected_cost_distance_final = torch.mean(cost_distance_final, 0)
    #
    #     else:
    #         raise NotImplementedError
    #
    #     return expected_cost_distance_final

    def starting_positions(self, n_samples):
        initial_positions = []
        for i in range(self.n_samples):
            initial_positions.append(
                torch.tensor(self.potentials[i].reporter.latest_positions))
        initial_positions = torch.stack(initial_positions).to(self.device)
        initial_positions = initial_positions.view(n_samples, -1)

        # Add static velocities
        init_velocities = torch.zeros_like(initial_positions)
        stacked = torch.cat([initial_positions, init_velocities], dim=1)

        return stacked

    def reset(self):
        for i in range(self.n_samples):
            self.potentials[i].reset()