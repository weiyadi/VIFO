import torch

import metrics
import data
from helpers import prepare_model, train_epoch_standard, eval_epoch_standard, eval_epoch_swag
import hydra
from omegaconf import DictConfig
import os

from posteriors import SWAG
import swag_utils


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/swag/{str(cfg.posterior)}'
    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs
    regression = cfg.basic.regression

    # use a slightly modified loss function that allows input of model
    if cfg.posterior.loss == 'CE':
        criterion = metrics.cross_entropy
        # criterion = F.cross_entropy
    elif cfg.poterior.loss == 'adv_CE':
        criterion = metrics.adversarial_cross_entropy
    elif cfg.posterior.loss == "NLL":
        criterion = metrics.nll
    else:
        raise NotImplementedError

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type)
    train_loader, _, test_loader = data.getDataloader(trainset, valset, cfg.basic.valid_size, cfg.basic.batch_size,
                                                    cfg.basic.num_workers, split_train=True)
    
    # optimizer settings
    lr_init = cfg.posterior.lr_init
    momentum = cfg.posterior.momentum
    wd = cfg.posterior.wd
    swag_lr = cfg.posterior.swag_lr

    # swag settings
    swag_start = cfg.posterior.swag_start
    L_test = cfg.posterior.L_test
    max_num_models = cfg.posterior.max_num_models

    model_args = list()
    model = model_cfg.base(*model_args, **model_kwargs).to(device)
    pretrain = cfg.posterior.get('pretrain', False)
    if pretrain:
        adam_pretrain = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        adam_pretrain = None

    swag_model = SWAG(model_cfg.base, subspace_type='covariance', subspace_kwargs={'max_rank': max_num_models},
                    *model_args, **model_kwargs)
    swag_model.to(device)
    
    def schedule(epoch):
        t = epoch / swag_start
        lr_ratio = swag_lr / lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return lr_init * factor

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr_init,
        momentum=momentum,
        weight_decay=wd
    )
    
    for epoch in range(epochs):
        if not cfg.posterior.no_schedule:
            lr = schedule(epoch)
            swag_utils.adjust_learning_rate(optimizer, lr)
        else:
            lr = lr_init
        if epoch < 5 and pretrain:
            train_loss, train_acc, _ = train_epoch_standard(train_loader, model, criterion, adam_pretrain,
                                                        regression=regression)
        else:
            train_loss, train_acc, _ = train_epoch_standard(train_loader, model, criterion, optimizer,
                                                        regression=regression)

        test_loss, test_acc = eval_epoch_standard(test_loader, model, criterion, regression=regression)

        if epoch + 1 > swag_start:
            swag_model.collect_model(model)
            swag_model.set_swa()
            swag_utils.bn_update(train_loader, swag_model)
            swa_loss, swa_acc = eval_epoch_standard(test_loader, swag_model, criterion, regression=regression)
            swag_loss, swag_acc = eval_epoch_swag(test_loader, swag_model, criterion, L_test, regression=regression)
        else:
            swa_loss = swa_acc = swag_loss = swag_acc = None

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \t'.format(
                epoch, train_loss, train_acc, test_loss, test_acc))
        if epoch + 1 > swag_start:
            print('SWA Loss: {:.4f} \tSWA Accuracy: {:.4f} \tSWAG Loss: {:.4f} \tSWAG Accuracy: {:.4f} \t'.format(
                swa_loss, swa_acc, swag_loss, swag_acc))


        torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')
        torch.save(swag_model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_swag.pt')

if __name__ == '__main__':
    main()
