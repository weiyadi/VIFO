import os
import torch

import metrics
import data
from helpers import train_epoch_standard, eval_epoch_standard, prepare_model
import hydra
from omegaconf import DictConfig

def neg_ll_log_prior(model, prior_var):
    # negative last layer log prior
    W = list(model.parameters())[-1]
    return 0.5 * torch.sum(W ** 2) / prior_var

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/laplace/{str(cfg.posterior)}/'
    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    regression = cfg.basic.regression
    net_type = cfg.net_type

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, net_type)
    train_loader, _, test_loader = data.getDataloader(trainset, valset, cfg.posterior.valid_size,
                                                            cfg.basic.batch_size, cfg.basic.num_workers,
                                                            split_train=True)
    prior_var = cfg.posterior.prior_var

    model_args = list()
    model = model_cfg.base(*model_args, **model_kwargs).to(device)

    if not regression:
        criterion = metrics.cross_entropy
    else:
        criterion = metrics.nll

    reg = lambda model: neg_ll_log_prior(model, prior_var) / len(trainset)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(epochs):
        train_loss, train_acc, neg_log_prior = train_epoch_standard(train_loader, model, criterion, optimizer,
                                                    regression=regression, regularization=reg)

        test_loss, test_acc = eval_epoch_standard(test_loader, model, criterion, regression=regression)
        
        print(f'Epoch {epoch}: train_loss {train_loss}, test_loss {test_loss}, train_acc {train_acc}, test_acc {test_acc}, neg_log_prior {neg_log_prior}')

    torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')

if __name__ == '__main__':
    main()
