import numpy as np
from torch import nn
import torch
import math

from torch.nn import functional as F

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def outputs_to_log_probs_classfication(outputs, aggregate='mean'):
    if outputs.dim() == 2:
        # no ensemble
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs
    else:
        log_probs = F.log_softmax(outputs, dim=2)
        if aggregate == 'mean':
            log_probs_aggregate = torch.mean(log_probs, dim=0)
        elif aggregate == 'logmeanexp':
            log_probs_aggregate = logmeanexp(log_probs, dim=0)
        else:
            return log_probs
        return log_probs_aggregate


def nll_regression(outputs, labels, aggregate='mean'):
    if outputs.dim() == 2:
        mean = outputs[:, 0]
        noise = torch.clamp(F.softplus(outputs[:, 1]), min=1e-3, max=1e3)
        nll = 0.5 * ((labels.squeeze() - mean) ** 2 / noise + torch.log(noise) + math.log(2 * math.pi))
        return torch.mean(nll, 0)
    else:
        mean = outputs[:, :, 0]
        noise = torch.clamp(F.softplus(outputs[:, :, 1]), min=1e-3, max=1e3)
        nll = 0.5 * ((labels.squeeze()[None, :] - mean) ** 2 / noise + torch.log(noise) + math.log(2 * math.pi))
        if aggregate == 'mean':
            nll_aggregate = torch.mean(nll, dim=0)
        elif aggregate == 'logmeanexp':
            nll_aggregate = -logmeanexp(-nll, dim=0)
        else:
            raise NotImplementedError
        return nll_aggregate


class ELBO(nn.Module):
    def __init__(self, train_size, smooth=None, normalize=True):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.smooth = smooth
        self.normalize = normalize

    def forward(self, inputs, target, kl, beta, regression=False, *args, **kwargs):
        assert not target.requires_grad
        if self.normalize:
            kl = kl / self.train_size
        if not regression:
            log_probs = outputs_to_log_probs_classfication(inputs, aggregate='mean')
            if self.smooth:
                log_probs = torch.log((1 - self.smooth) * torch.exp(log_probs) + self.smooth)
            return F.nll_loss(log_probs, target, reduction='mean') + beta * kl
        else:
            # regression
            nll = nll_regression(inputs, target, aggregate='mean')
            if self.smooth:
                nll = -torch.log((1 - self.smooth) * torch.exp(-nll) + self.smooth)
            return torch.mean(nll, dim=0) + beta * kl


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def cross_entropy(model, input, target, *args, **kwargs):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output, {}


def dir_elbo(model, input, target, *args, **kwargs):
    beta = kwargs.get('beta', 1.0)
    output = model(input)
    dir_alpha = F.relu(output) + 1
    N = input.size(0)
    sum_dir_alpha = torch.sum(dir_alpha, dim=1, keepdim=True)
    loss = torch.digamma(sum_dir_alpha) - torch.digamma(dir_alpha[torch.arange(N), target]).unsqueeze(-1)

    # compute the kl divergence
    kl = torch.lgamma(sum_dir_alpha) - torch.lgamma(torch.tensor(output.size(1), device=input.device)) - torch.sum(
        torch.lgamma(dir_alpha), dim=1, keepdim=True) + torch.sum(
        (dir_alpha - 1) * (torch.digamma(dir_alpha) - torch.digamma(sum_dir_alpha)), dim=1, keepdim=True)
    return torch.mean(loss + beta * kl), output, {'kl': torch.mean(kl)}


def dir_loss(model, input, target, *args, **kwargs):
    output = model(input)
    dir_alpha = F.relu(output) + 1
    N = input.size(0)
    sum_dir_alpha = torch.sum(dir_alpha, dim=1, keepdim=True)
    loss = torch.log(sum_dir_alpha) - torch.log(dir_alpha[torch.arange(N), target]).unsqueeze(-1)
    return torch.mean(loss), output, {}


def kl_div(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.kl_div(F.log_softmax(output), target, reduction="batchmean")

    return loss, output, {}


def cross_entropy_output(output, target):
    # standard cross-entropy loss function

    loss = F.cross_entropy(output, target)

    return loss, {}


def masked_loss(y_pred, y_true, void_class=11., weight=None, reduce=True):
    # masked version of crossentropy loss

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction='none')
    loss = mask.float() * loss

    if reduce:
        return loss.sum() / mask.sum()
    else:
        return loss, mask


def nll(model, input, target):
    output = model(input)
    nll = nll_regression(output, target)
    return torch.mean(nll), output, {}

