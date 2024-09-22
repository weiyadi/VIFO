import torch

import metrics
import data
from helpers import prepare_model, train_epoch_standard, eval_epoch_standard
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
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/dir/{str(cfg.posterior)}'
    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    regression = cfg.basic.regression

    if cfg.posterior.loss == 'dir_elbo':
        criterion = lambda model, inputs, targets: metrics.dir_elbo(model, inputs, targets,
                                                                    beta=cfg.posterior.reg_param)
    else:
        raise NotImplementedError

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type)
    train_loader, _, test_loader = data.getDataloader(trainset, valset, cfg.basic.valid_size, cfg.basic.batch_size,
                                                    cfg.basic.num_workers, split_train=True)

    model_args = list()
    model = model_cfg.base(*model_args, **model_kwargs).to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        train_obj, train_acc, train_kl = train_epoch_standard(train_loader, model, criterion, optimizer,
                                                            regression=regression)

        test_loss, test_acc = eval_epoch_standard(test_loader, model, metrics.dir_loss, regression=regression)

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_obj, train_acc, test_loss, test_acc, train_kl))


        torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')


if __name__ == '__main__':
    main()
