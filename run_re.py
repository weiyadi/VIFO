import torch

import data
from repulsive_ensembles import f_WGD, Ensemble, RBF, SpectralSteinEstimator
from helpers import prepare_model
import hydra
from omegaconf import DictConfig
import logging
import os
from torch.distributions.normal import Normal
import torch.nn.functional as F
import math
from metrics import logmeanexp
import metrics
from swag_utils import flatten

log = logging.getLogger(__name__)

def train_epoch_re(loader, particles, f_wgd, device=None, step=0):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        f_wgd.step(particles, inputs, labels, step=step)

    return particles

def eval_epoch_re(loader, ensemble, W, device=None, regression=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_nll = 0.
    total_acc = 0.
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = ensemble.forward(inputs, W) # n_particles x n_batch x n_classes for classification or n_particles x n_batch x 2 for regression
        if regression:
            nll = metrics.nll_regression(output, labels, aggregate='logmeanexp').mean(0)
            acc = metrics.rmse(output, labels)
        else:
            log_pred = F.log_softmax(output, dim=2)
            log_pred_avg = logmeanexp(log_pred, dim=0)
            nll = F.nll_loss(log_pred_avg, labels, reduction='mean')
            acc = metrics.acc(log_pred_avg.detach(), labels)
        total_nll += nll.item()
        total_acc += acc.item()
    return total_nll / len(loader), total_acc / len(loader)


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/re/{str(cfg.posterior)}/'
    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    uci_regression = cfg.basic.regression
    normalize = cfg.basic.get('normalize', False)

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type, uci_regression=uci_regression, normalize=normalize)
    train_loader, val_loader, test_loader = data.getDataloader(trainset, valset, cfg.posterior.valid_size,
                                                            cfg.basic.batch_size, cfg.basic.num_workers,
                                                            split_train=True)

    model_args = list()
    ensemble = Ensemble(device, model_cfg.base, *model_args, **model_kwargs)
    W = (1 * torch.randn(cfg.posterior.n_particles, ensemble.rank)).to(device)
    # He initialization
    for i in range(cfg.posterior.n_particles):
        net = model_cfg.base(*model_args, **model_kwargs)
        [p.requires_grad_(False) for p in net.parameters()]
        W[i] = flatten(list(net.parameters()))
    W = W.requires_grad_(True)

    # speicify prior
    prior = Normal(0., math.sqrt(cfg.posterior.prior_var))
    
    # specify ssge
    ssge_k = RBF() # sigma is None, use median strategy
    ssge = SpectralSteinEstimator(0.9, None, ssge_k, None)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=[W])

    # specify f_wgd
    annealing_schedule = [1 for _ in range(epochs)]
    f_wgd = f_WGD(ensemble, prior, ssge_k, optimizer, annealing_schedule, ssge, num_train=False, method=cfg.posterior.method, device=device, regression=uci_regression)

    for epoch in range(epochs):
        train_epoch_re(train_loader, W, f_wgd, device, step=epoch)
        train_loss, train_acc = eval_epoch_re(train_loader, ensemble, W, device, regression=uci_regression)
        test_loss, test_acc = eval_epoch_re(test_loader, ensemble, W, device, regression=uci_regression)
        print(f'Epoch {epoch}: train_loss {train_loss}, test_loss {test_loss}, train_acc {train_acc}, test_acc {test_acc}')

    torch.save(W, f'{save_dir}/seed{cfg.basic.seed}_model.pt')


if __name__ == '__main__':
    main()
