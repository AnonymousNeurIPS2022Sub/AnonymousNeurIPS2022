import torch
from torch import nn


class NNPolicy(nn.Module):
    def __init__(self, device, force, dims, T, bridge = False):
        super().__init__()
        self.device = device
        self.force = force
        self.T = T
        self.bridge = bridge
        self.dims = dims

        self.input_dims = self.dims
        if self.bridge:
            self.input_dims += 1

        if not self.force:
            self.output_dim = 1
            self.inc_last_bias = False
        else:
            self.output_dim = self.dims
            self.inc_last_bias = True


        self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_dims, 512), # M x (3 x 2)
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.output_dim, bias=self.inc_last_bias)
            )

        self.to(self.device)

    def forward(self, x_in, t):
        x = x_in.clone()
        x = x[:, :int(x.shape[1] / 2)].view(x.shape[0], -1)

        if not self.force:
            x.requires_grad = True

        x_ = x

        if self.bridge: # We need to include the time step if we bridge
            tmp = t.unsqueeze(0).repeat(x_.shape[0], 1).to(x_.device) / self.T
            x_ = torch.hstack([x_, tmp])


        if not self.force:
            energy = self.linear_relu_stack(x_) * 50
            force = -torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=torch.ones_like(energy), create_graph=True, retain_graph=True)[0]
            return energy, force
        else:
            force = self.linear_relu_stack(x_)
            force = force.view(-1, self.dims)
            return force
