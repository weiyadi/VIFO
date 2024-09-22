# Modified from https://github.com/izmailovpavel/understandingbdl/blob/master/swag/posteriors/vi_model.py
import torch
import math
from posteriors.posteriors_utils import kl, collapsed_kl_mean, collapsed_kl_mv, eb_kl, set_weights

class VIFFGModel(torch.nn.Module):
    def __init__(self, base, priors, reg, cfg, pos_mean=None, pos_rho=None, eps=1e-6, *args, **kwargs):
        super(VIFFGModel, self).__init__()

        self.base = base
        self.args = args
        self.kwargs = kwargs
        model = self.base(*args, **kwargs)
        self.rank = sum([param.numel() for param in model.parameters()])
        self.eps = eps

        self.prior_mu = torch.tensor(priors['prior_mu'])
        self.prior_log_sigma = torch.tensor(math.log(priors['prior_sigma']))
        if pos_mean is None:
            mu_mean, mu_sig = priors['posterior_mu_initial']
            self.mu = torch.nn.Parameter(torch.empty(self.rank).normal_(mu_mean, mu_sig))
        else:
            self.mu = torch.nn.Parameter(pos_mean)

        if pos_rho is None:
            rho_mean, rho_sig = priors['posterior_rho_initial']
            self.rho = torch.nn.Parameter(torch.empty(self.rank).normal_(rho_mean, rho_sig))
        else:
            self.rho = torch.nn.Parameter(pos_rho)

        self.setup_reg(reg, cfg)

    def setup_reg(self, reg, cfg):
        self.reg = reg
        if reg == 'mean':
            self.gamma = torch.tensor(cfg['gamma'])
            self.alpha_reg = torch.tensor(cfg['alpha_reg'])
            self.const = 0.5 * torch.log(self.gamma) - 0.5
        elif reg == 'mv':
            self.alpha = torch.tensor(cfg['alpha'])
            self.beta = torch.tensor(cfg['beta'])
            self.delta = torch.tensor(cfg['delta'])
            self.const = -torch.lgamma(self.alpha + 0.5) + torch.lgamma(self.alpha) - self.alpha * torch.log(
                self.beta) - torch.log(self.delta)
        elif reg == 'eb':
            self.gamma_alpha = torch.tensor(cfg['gamma_alpha'])
            self.gamma_beta = torch.tensor(cfg['gamma_beta'])

    def forward(self, L, *args, **kwargs):
        device = self.rho.device
        sigma = torch.nn.functional.softplus(self.rho)
        sigma = torch.clamp(sigma, min=self.eps)

        if L > 0:
            outputs = []
            for _ in range(L):
                w = self.mu + torch.randn(self.rank, device=device) * sigma
                model = self.base(*self.args, **self.kwargs).to(device)
                [p.requires_grad_(False) for p in model.parameters()]
                set_weights(model, w, device)
                output = model(*args, **kwargs)
                outputs.append(output)
            return torch.stack(outputs)
        else:
            model = self.base(*self.args, **self.kwargs).to(device)
            [p.requires_grad_(False) for p in model.parameters()]
            set_weights(model, self.mu, device)
            output = model(*args, **kwargs)
            return output

    def sample(self):
        sigma = torch.nn.functional.softplus(self.rho)
        sigma = torch.clamp(sigma, self.eps)
        w = torch.randn(self.rank) * sigma + self.mu
        return w

    def compute_kl(self, *args, **kwargs):
        print("deprecated KL")
        sigma = torch.nn.functional.softplus(self.rho)
        sigma = torch.clamp(sigma, min=self.eps)
        prior_sigma = torch.exp(self.prior_log_sigma)

        kl_val = kl(self.mu, sigma, self.prior_mu, prior_sigma, log_prior_sigma=self.prior_log_sigma)
        return kl_val

    def compute_reg(self, *args, **kwargs):
        sigma = torch.nn.functional.softplus(self.rho)
        sigma = torch.clamp(sigma, min=self.eps)
        prior_sigma = torch.exp(self.prior_log_sigma)
        log_sigma = torch.log(sigma)
        if self.reg == 'naive':
            kl_val = kl(self.mu, sigma, self.prior_mu, prior_sigma, log_prior_sigma=self.prior_log_sigma)
            return kl_val
        elif self.reg == 'mean':
            kl_val = collapsed_kl_mean(self.mu, sigma, self.gamma, self.alpha_reg, log_sigma=log_sigma,
                                       const=self.const)
            return kl_val
        elif self.reg == 'mv':
            kl_val = collapsed_kl_mv(self.mu, sigma, self.alpha, self.beta, self.delta, log_sigma=log_sigma,
                                     const=self.const)
            return kl_val
        elif self.reg == 'eb':
            kl_val = eb_kl(self.mu, sigma, self.gamma_alpha, self.gamma_beta)
            return torch.sum(kl_val)
        else:
            raise NotImplementedError

    def compute_entropy(self):
        sigma = torch.nn.functional.softplus(self.rho) + self.eps
        return torch.sum(torch.log(sigma))
