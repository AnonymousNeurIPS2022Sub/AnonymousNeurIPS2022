import os

import numpy as np
import torch

from Dynamics.MoleculeBase import MoleculeBaseDynamics
from potentials.chignolin_md import ChignolinPotentialMD


class ChignolinDynamics(MoleculeBaseDynamics):
    def __init__(self, loss_func, n_samples=10, device='cpu', bridge=False, save_file=None, bridge_samples = 128):
        super().__init__(loss_func, n_samples, device, bridge, save_file, bridge_samples)

    def _init_ending_positions(self):
        overide = True
        if self.bridge:
            n = self.bridge_samples
            path = './potentials/files/target_chignolin.npy'
        else:
            n = 1
            path = './potentials/files/target_chignolin.npy'

        if os.path.exists(path) and not overide:
            print("Existing target points exits, loading them")
            positions = np.load(path)
            ending_positions = torch.as_tensor(positions)

        else:
            print("Generating target points")
            positions = []
            pot = ChignolinPotentialMD('./potentials/files/chignolin_folded.pdb', -1)
            pot.simulation.minimizeEnergy()
            pot.simulation.step(1)
            for i in range(n):
                print(f"{i} of {n}")
                if i > 0:
                    pot.simulation.step(500)
                end_positions = torch.tensor(pot.reporter.latest_positions)
                positions.append(end_positions.clone())

            ending_positions = torch.stack(positions)
            np.save(path, ending_positions.detach().cpu().numpy())

        return ending_positions

    def _init_potentials(self):
        # Initialize potentials
        potentials = []
        for i in range(self.n_samples):
            pot = ChignolinPotentialMD('./potentials/files/chignolin_unfolded.pdb', i, bridge=self.bridge, save_file=self.save_file)
            potentials.append(pot)

        return potentials


    def phi(self, x):
        end_ = self.ending_positions.view(self.ending_positions.shape[0], -1)
        x_ = x[:, :int(x.shape[1] / 2)]

        if self.loss_func == 'pairwise_dist':
            x_ = x_.view(x_.shape[0], -1, 3)
            end_ = end_.view(end_.shape[0], -1, 3)
            px = torch.cdist(x_, x_).unsqueeze(0)
            pend = torch.cdist(end_, end_).unsqueeze(1).repeat(1,
                                                               self.n_samples,
                                                               1, 1)

            t = (px - pend) ** 2
            cost_distance = torch.mean(t, dim=(2, 3))

            cost_distance_final = (cost_distance).exp() * 10 #, * 50, * 100

            expected_cost_distance_final = torch.mean(cost_distance_final, 0)

        else:
            raise NotImplementedError

        return expected_cost_distance_final
