import numpy as np
import torch

def set_weights(model, vector, device=None):
    offset = 0
    for name, param in model.named_parameters():
        param.copy_(vector[offset:(offset + param.numel())].view(param.size()).to(device))
        offset += param.numel()

class Ensemble():
    """Implementation of an ensemble of models

    This is a simple class to manage and make predictions using an ensemble with or without particles
    Args:
        device: Torch device (cpu or gpu).
        net: pytorch model to create the ensemble
        particles(Tensor): Tensor (n_particles, n_params) containing squeezed parameter value of the specified model,
            if None, particles will be sample from a gaussian N(0,1)
        n_particles(int): if no particles are provided the ensemble is initialized and the number of members is required

    """

    def __init__(self, device, base, *args, **kwargs):
        self.base = base
        self.args = args
        self.kwargs = kwargs
        model = base(*args, **kwargs).to(device)
        self.rank = sum([param.numel() for param in model.parameters()])
        self.device = device

    # def reshape_particles(self, z):
    #     reshaped_weights = []
    #     z_splitted = torch.split(z, self.weighs_split, 1)
    #     for j in range(z.shape[0]):
    #         l = []
    #         for i, shape in enumerate(self.net.param_shapes):
    #             l.append(z_splitted[i][j].reshape(shape))
    #         reshaped_weights.append(l)
    #     return reshaped_weights

    def forward(self, x, W):
        output = []
        for i in range(W.size(0)):
            model = self.base(*self.args, **self.kwargs).to(self.device)
            [p.requires_grad_(False) for p in model.parameters()]
            set_weights(model, W[i], self.device)
            output.append(model(x))
        return torch.stack(output)
        # models = self.reshape_particles(W)
        # if self.net.out_act is None:
        #     pred = [self.net.forward(x, w) for w in models]
        #     return [torch.stack(pred)] #.unsqueeze(0)
        # else:
        #     pred,hidden = zip(*(list(self.net.forward(x,w)) for w in models))
        #     return torch.stack(pred), torch.stack(hidden)