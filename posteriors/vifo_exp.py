import torch
from .vifo import VIFOBase


class VIFOSingleExp(torch.nn.Module, VIFOBase):

    def __init__(self, base, prior_var, reg, cfg, eps=1e-6, *args, **kwargs):
        super(VIFOSingleExp, self).__init__()

        self.base = base
        if 'num_classes' in kwargs:
            self.num_classes = kwargs['num_classes']
            kwargs['num_classes'] = 2 * self.num_classes
        else:
            self.num_classes = 2
            kwargs['num_classes'] = 4
        self.base_model = base(*args, **kwargs)
        self.prior_var = torch.tensor(prior_var)
        self.eps = eps
        self.setup_reg(reg, cfg)

    def get_mean_sigma(self, *args, **kwargs):
        val = self.base_model(*args, **kwargs)
        mean = val[:, :self.num_classes]
        sigma = torch.exp(val[:, self.num_classes:])
        sigma = torch.clamp(sigma, min=self.eps)
        return mean, sigma

    def forward(self, L, *args, **kwargs):
        reg = kwargs.pop('reg', False)
        mean, sigma = self.get_mean_sigma(*args, **kwargs)
        eps = torch.randn(L, mean.size(0), mean.size(1), device=mean.device)
        output = mean[None, :] + eps * sigma[None, :]
        if not reg:
            return output
        else:
            return output, self.compute_reg(mean, sigma)

    def compute_reg(self, mean, sigma, *args, **kwargs):
        if 'l2' not in self.reg:
            return super().compute_reg(mean, sigma, *args, **kwargs)
        elif self.reg == 'l2':
            norm = 0.
            for param in self.parameters():
                norm += torch.sum(param.pow(2))
            return norm
        else:
            raise NotImplementedError
