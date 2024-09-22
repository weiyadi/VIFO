import torch
from torch.nn import functional as F
from posteriors.posteriors_utils import kl, collapsed_kl_mean, collapsed_kl_mv, eb_kl, collapsed_kl_mean_all, \
    collapsed_kl_mv_all, eb_kl_all


class VIFOBase:
    def __init__(self):
        pass

    def setup_reg(self, reg, cfg):
        self.reg = reg
        self.kl_mode = 'sum'
        if reg == 'mean':
            self.gamma = torch.tensor(cfg['gamma'])
            self.alpha_reg = torch.tensor(cfg['alpha_reg'])
            self.const = 0.5 * torch.log(self.gamma) - 0.5
        elif 'mv' in reg:
            self.alpha = torch.tensor(cfg['alpha'])
            self.beta = torch.tensor(cfg['beta'])
            self.delta = torch.tensor(cfg['delta'])
            self.const = -torch.lgamma(self.alpha + 0.5) + torch.lgamma(self.alpha) - self.alpha * torch.log(
                self.beta) - torch.log(self.delta)
        elif 'eb' in reg:
            self.gamma_alpha = torch.tensor(cfg['gamma_alpha'])
            self.gamma_beta = torch.tensor(cfg['gamma_beta'])
        elif reg == 'mean_all':
            self.gamma = torch.tensor(cfg['gamma'])
            self.alpha_reg = torch.tensor(cfg['alpha_reg'])

    def compute_reg(self, mean, sigma, *args, **kwargs):
        log_sigma = torch.log(sigma)
        if self.reg == 'naive':
            log_prior_sigma = 0.5 * torch.log(self.prior_var)
            kl_val = kl(mean, sigma, 0., torch.sqrt(self.prior_var), log_prior_sigma=log_prior_sigma,
                        kl_mode=self.kl_mode)
            return kl_val / mean.size(0)
        elif self.reg == 'mean':
            kl_val = collapsed_kl_mean(mean, sigma, self.gamma, self.alpha_reg, log_sigma=log_sigma, const=self.const,
                                       kl_mode=self.kl_mode)
            return kl_val / mean.size(0)
        elif self.reg == 'mv':
            kl_val = collapsed_kl_mv(mean, sigma, self.alpha, self.beta, self.delta, log_sigma=log_sigma,
                                     const=self.const, kl_mode=self.kl_mode)
            return kl_val / mean.size(0)
        elif self.reg == 'eb':
            kl_val = eb_kl(mean, sigma, self.gamma_alpha, self.gamma_beta, kl_mode=self.kl_mode)
            return torch.sum(kl_val) / mean.size(0)
        elif self.reg == 'mean_all':
            kl_val = collapsed_kl_mean_all(mean, sigma, self.gamma, self.alpha_reg, log_sigma=log_sigma)
            return kl_val / mean.size(0)
        elif self.reg == 'mv_all':
            kl_val = collapsed_kl_mv_all(mean, sigma, self.alpha, self.beta, self.delta, log_sigma=log_sigma)
            return kl_val / mean.size(0)
        elif self.reg == 'eb_all':
            kl_val = eb_kl_all(mean, sigma, self.gamma_alpha, self.gamma_beta)
            return torch.sum(kl_val) / mean.size(0)
        else:
            raise NotImplementedError


class VIFO(torch.nn.Module, VIFOBase):

    def __init__(self, base, prior_var, reg, cfg, eps=1e-6, *args, **kwargs):
        super(VIFO, self).__init__()

        self.base = base
        self.mean_model = base(*args, **kwargs)
        self.sigma_model = base(*args, **kwargs)
        self.prior_var = torch.tensor(prior_var)
        self.eps = eps
        self.setup_reg(reg, cfg)

    def get_mean_sigma(self, *args, **kwargs):
        mean = self.mean_model(*args, **kwargs)
        sigma = F.softplus(self.sigma_model(*args, **kwargs))
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
        elif self.reg == 'l2log':
            norm = 0.
            for param in self.parameters():
                norm += torch.sum(param.pow(2))
            log_term = 0.
            for param in self.sigma_model.parameters():
                log_term += torch.sum(torch.log(param ** 2))
            return 0.5 / self.prior_var * norm - 0.5 * log_term


class VIFOSingle(torch.nn.Module, VIFOBase):

    def __init__(self, base, prior_var, reg, cfg, kl_mode='sum', eps=1e-6, *args, **kwargs):
        super(VIFOSingle, self).__init__()

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
        self.kl_mode = kl_mode
        self.setup_reg(reg, cfg)

    def get_mean_sigma(self, *args, **kwargs):
        val = self.base_model(*args, **kwargs)
        mean = val[:, :self.num_classes]
        sigma = F.softplus(val[:, self.num_classes:])
        sigma = torch.clamp(sigma, min=self.eps)
        return mean, sigma

    def forward(self, L, *args, **kwargs):
        reg = kwargs.pop('reg', False)
        mean, sigma = self.get_mean_sigma(*args, **kwargs)
        if L > 0:
            eps = torch.randn(L, mean.size(0), mean.size(1), device=mean.device)
            output = mean[None, :] + eps * sigma[None, :]
            if not reg:
                return output
            else:
                return output, self.compute_reg(mean, sigma)
        else:
            return mean

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
