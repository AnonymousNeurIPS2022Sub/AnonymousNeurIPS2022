import os
import time

import einops as einops
import numpy as np
import torch
import torch.nn as nn
import shutil

from Dynamics.Alanine import AlanineDynamics
from Dynamics.ChignolinDynamics import ChignolinDynamics
from Dynamics.Poly import PolyDynamics
from plotting.BasePlotter import AfterEpochPlotter
from plotting.Loggers import CostsLogger
from solvers.PICE import PICE

from egnn_utils import EGNN
from egnn_utils import prepare_data


class MoleculeLogger(AfterEpochPlotter):
    def __init__(self, file, interval=1):
        super().__init__(file, interval)
        self.file = file
        self.paths = []
        np.set_printoptions(linewidth=np.inf)

    def plot(self, paths, run, us):
        np.save(f'{self.file}/{run}_paths', paths.detach().cpu().numpy())



def create_timed_folder(file):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    timed_file = f"{file}/{timestr}"
    if not os.path.exists(timed_file):
        os.makedirs(timed_file)
    return timed_file


if __name__ == "__main__":
    # Set Seed
    torch.manual_seed(42)

    # File setup
    file = "./results/Alanine"
    file = create_timed_folder(file)

    # Copy important files
    current_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(f"{file}/code")
    shutil.copytree(f"{current_path}/plotting", f"{file}/code/plotting")
    shutil.copytree(f"{current_path}/potentials",
                             f"{file}/code/potentials")
    shutil.copytree(f"{current_path}/solvers",
                             f"{file}/code/solvers")
    shutil.copyfile(f"{current_path}/MoleculeExperiment.py",
                             f"{file}/code/MoleculeExperiment.py")


    policy = 'mlp'
    force = False # if false -> use energy
    bridge = False

    device = 'cuda'

    T = 5000.
    dt = torch.tensor(1.)
    n_steps = int(T / dt)

    n_rollouts = 10000
    n_samples = 16

    lr = 0.0001 # 0.00001

    # environment = AlanineDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, bridge=bridge, save_file=file)
    # environment = ChignolinDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, bridge=bridge, save_file=file)
    environment = PolyDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, bridge=bridge, save_file=file)
    from policies.Poly import NNPolicy
    dims = environment.dims

    std = torch.tensor(.1).to(device) # .05
    # std = torch.tensor(.1).to(device)  # EGNN
    R = torch.eye(dims).to(device)

    logger = CostsLogger(f'{file}')
    plotter = MoleculeLogger(file)
    targets = environment.ending_positions

    start_plot = environment.starting_positions(n_samples)
    # start_plot = start_plot[:, :int(start_plot.shape[1]/2)]
    plotter.plot(torch.stack([start_plot, targets.view(1, -1).repeat(n_samples, 2)], axis=1), 'start', None)

    np.save(f"{file}/target", targets.detach().cpu().numpy())

    # Simple NN policy
    # torch.autograd.set_detect_anomaly(True)
    if policy == 'mlp':
        nn_policy = NNPolicy(device, dims = dims, force=force, T=T, bridge=bridge)
    elif policy == 'egnn':
        n_nodes = 22
        n_nodes, edge_attr, edges, h0 = prepare_data(n_samples)
        nn_policy = EGNN(in_node_nf=4, in_edge_nf=0, hidden_nf=32, n_nodes=n_nodes, edge_attr=edge_attr,
        edges=edges, h0=h0, n_layers=4, coords_weight=1.0, attention=True, node_attr=0, force=force, device=device)
    else:
        raise ValueError

    PICE(environment, nn_policy, n_rollouts, n_samples, n_steps, dt, std, dims * 2, R, logger, [], True, file, device=device, lr=lr)

    torch.save(nn_policy, f'{file}/final_policy')




