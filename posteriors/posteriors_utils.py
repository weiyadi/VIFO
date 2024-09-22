import torch
import math

LOG2PI = math.log(2 * math.pi)

def kl(mu, sigma, prior_mu, prior_sigma, log_prior_sigma=None, kl_mode='sum'):
    if log_prior_sigma is None:
        log_prior_sigma = torch.log(prior_sigma)
    if kl_mode == 'sum':
        kl = 0.5 * (2 * (log_prior_sigma - torch.log(sigma)) - 1 + (sigma / prior_sigma).pow(2) + (
                (mu - prior_mu) / prior_sigma).pow(2)).sum()
        return kl
    else:
        kl = torch.sum(0.5 * (2 * (log_prior_sigma - torch.log(sigma)) - 1 + (sigma / prior_sigma).pow(2) + (
                (mu - prior_mu) / prior_sigma).pow(2)), 1)
        return torch.max(kl)


def collapsed_kl_mean(mu, sigma, gamma, alpha_reg, log_sigma=None, const=None, kl_mode='sum'):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    if const is None:
        const = 0.5 * torch.log(gamma) - 0.5
    if kl_mode == 'sum':
        result = 0.5 / gamma * (sigma ** 2 + alpha_reg * mu ** 2) - log_sigma - 0.5 * torch.log(alpha_reg) + const
        return torch.sum(result)
    else:
        result = 0.5 / gamma * (sigma ** 2 + alpha_reg * mu ** 2) - log_sigma - 0.5 * torch.log(alpha_reg) + const
        return torch.max(torch.sum(result, 1))


def collapsed_kl_mean_all(mu, sigma, gamma, alpha_reg, log_sigma=None):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    mu_avg = torch.sum(mu, 0) / mu.size(0)
    result = 0.5 / gamma * (sigma ** 2 + mu ** 2) - log_sigma
    const = -0.5 * (1 - alpha_reg) / gamma * torch.sum(mu_avg ** 2) + 0.5 * mu.size(1) * torch.log(
        gamma / alpha_reg) - 0.5 * mu.size(1)
    return torch.sum(result) + mu.size(0) * const


def collapsed_kl_mv(mu, sigma, alpha, beta, delta, log_sigma=None, const=None, kl_mode='sum'):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    if const is None:
        const = -torch.lgamma(alpha + 0.5) + torch.lgamma(alpha) - alpha * torch.log(beta) - torch.log(delta)
    result = (alpha + 0.5) * torch.log(beta + 0.5 * delta * mu ** 2 + 0.5 * sigma ** 2) - log_sigma + const
    if kl_mode == 'sum':
        return torch.sum(result)
    else:
        return torch.max(torch.sum(result, 1))


def collapsed_kl_mv_all(mu, sigma, alpha, beta, delta, log_sigma=None, const=None):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    if const is None:
        const = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) - alpha * torch.log(beta) - 0.5 * torch.log(delta)
    N, D = mu.size(0), mu.size(1)
    mu_bar_2 = torch.sum(mu ** 2, 0) / N
    sigma_bar_2 = torch.sum(sigma ** 2, 0) / N
    result = (alpha + 0.5) * torch.log(beta + 0.5 * delta * mu_bar_2 + 0.5 * sigma_bar_2) - log_sigma
    return torch.sum(result) + N * D * const


def eb_kl(mu, sigma, alpha, beta, kl_mode='sum'):
    if mu.dim() == 2:
        # uq part
        N = mu.size(1)
        s_star = (torch.sum(mu ** 2 + sigma ** 2, 1) + 2 * beta) / (N + 2 * alpha + 2)
        s_star = s_star.reshape(-1, 1)
    else:
        # vi part
        N = mu.numel()
        s_star = (torch.sum(mu ** 2 + sigma ** 2) + 2 * beta) / (N + 2 * alpha + 2)
    log_prior_sigma = 0.5 * torch.log(s_star)
    kl_val = kl(mu, sigma, 0, torch.sqrt(s_star), log_prior_sigma=log_prior_sigma, kl_mode=kl_mode)
    return kl_val


def eb_kl_all(mu, sigma, alpha, beta):
    mu_bar_2 = torch.sum(mu ** 2, 0) / mu.size(0)
    sigma_bar_2 = torch.sum(sigma ** 2, 0) / sigma.size(0)
    s_star = (torch.sum(mu_bar_2 ** 2 + sigma_bar_2 ** 2) + 2 * beta) / (mu.size(1) + 2 * alpha + 2)
    log_prior_sigma = 0.5 * torch.log(s_star)
    kl_val = kl(mu, sigma, 0, torch.sqrt(s_star), log_prior_sigma=log_prior_sigma)
    return torch.sum(kl_val)


def set_weights(model, vector, device=None):
    offset = 0
    for name, param in model.named_parameters():
        param.copy_(vector[offset:(offset + param.numel())].view(param.size()).to(device))
        offset += param.numel()


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

