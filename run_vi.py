import torch

import metrics
import data
from helpers import prepare_model, train_epoch_vi, eval_epoch_vi
import hydra
from omegaconf import DictConfig
import logging
import os
from posteriors import VIFFGModel
import math

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/vi/{str(cfg.posterior)}/'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)
    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    uci_regression = cfg.basic.regression

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type, uci_regression=uci_regression)
    train_loader, _, test_loader = data.getDataloader(trainset, valset, cfg.basic.valid_size, cfg.basic.batch_size,
                                                        cfg.basic.num_workers, split_train=True)

    model_args = list()
    priors = {'prior_mu': 0., 'prior_sigma': math.sqrt(cfg.posterior.prior_var), 'posterior_mu_initial': (0., 0.1),
                'posterior_rho_initial': (-5, 0.1)}
    model = VIFFGModel(model_cfg.base, priors, cfg.posterior.reg, cfg.posterior, *model_args, **model_kwargs).to(
        device)
    if cfg.posterior.metric == 'ELBO':
        criterion = metrics.ELBO(len(train_loader.dataset))
    elif cfg.posterior.metric == 'DLM':
        criterion = metrics.DLM(len(train_loader.dataset))
    else:
        raise NotImplementedError

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    L_train = cfg.posterior.L_train
    L_test = cfg.posterior.L_test
    beta = cfg.posterior.reg_param
    for epoch in range(epochs):
        train_obj, train_acc, train_kl = train_epoch_vi(model, optimizer, criterion, train_loader, beta=beta, L=L_train,
                                                regression=uci_regression, sample_kl=False)
        test_loss, test_acc = eval_epoch_vi(model, test_loader, L=L_test, regression=uci_regression)

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_obj, train_acc, test_loss, test_acc, train_kl))

    torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')


if __name__ == '__main__':
    main()
