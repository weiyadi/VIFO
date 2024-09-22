import torch

import metrics
import data
from posteriors import VIFOSingle, VIFO
from helpers import prepare_model, train_epoch_vifo, eval_epoch_vi
import hydra
from omegaconf import DictConfig
import logging
import os

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/uq/{str(cfg.posterior)}/'
    os.makedirs(save_dir, exist_ok=True)


    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    regression = cfg.basic.regression

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type)
    train_loader, val_loader, test_loader = data.getDataloader(trainset, valset, cfg.posterior.valid_size,
                                                            cfg.basic.batch_size, cfg.basic.num_workers,
                                                            split_train=True)

    model_args = list()
    if cfg.posterior.single:
        model = VIFOSingle(model_cfg.base, cfg.posterior.prior_var, cfg.posterior.reg_type, cfg.posterior.vi, *model_args, **model_kwargs).to(device)
    else:
        model = VIFO(model_cfg.base, cfg.posterior.prior_var, cfg.posterior.reg_type, cfg.posterior.vi, *model_args, **model_kwargs).to(device)

    if cfg.posterior.aggregate == 'mean':
        criterion = metrics.ELBO(len(train_loader.dataset), normalize=False)
    elif cfg.posterior.aggregate == 'logmeanexp':
        criterion = metrics.DLM(len(train_loader.dataset), normalize=False)
    else:
        raise NotImplementedError

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = None

    L_train = cfg.posterior.L_train
    L_test = cfg.posterior.L_test
    beta = cfg.posterior.reg_param
    beta_ood = cfg.posterior.reg_param_ood
    for epoch in range(epochs):
        train_obj, train_acc, _, _ = train_epoch_vifo(model, optimizer, criterion, train_loader, beta=beta,
                                                L=L_train, regression=regression, scheduler=scheduler, ood='uniform', beta_ood=beta_ood)
        test_loss, test_acc = eval_epoch_vi(model, test_loader, L=L_test, regression=regression)
        print("Epoch", epoch, "train_obj", train_obj, "train_acc", train_acc, "test_loss", test_loss, "test_acc", test_acc)

    torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')


if __name__ == '__main__':
    main()
