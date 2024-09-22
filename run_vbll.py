import torch

import data
from posteriors import DiscClassification
from helpers import prepare_model
import hydra
from omegaconf import DictConfig
import logging
import os
import numpy as np

log = logging.getLogger(__name__)

def eval_acc(preds, y):
    map_preds = torch.argmax(preds, dim=1)
    return (map_preds == y).float().mean()

def train_epoch(model, optimizer, dataloader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    running_loss = []
    running_acc = []

    for train_step, data in enumerate(dataloader):
        optimizer.zero_grad()
        x = data[0].to(device)
        y = data[1].to(device)

        out = model(x)

        loss = out.train_loss_fn(y)
        probs = out.predictive.probs
        acc = eval_acc(probs, y).item()

        running_loss.append(loss.item())
        running_acc.append(acc)

        loss.backward()
        optimizer.step()

    return np.mean(running_loss), np.mean(running_acc)

def eval_epoch(model, dataloader, device=None):
    running_val_loss = []
    running_val_acc = []

    with torch.no_grad():
        model.eval()
        for test_step, data in enumerate(dataloader):
            x = data[0].to(device)
            y = data[1].to(device)

            out = model(x)
            loss = out.val_loss_fn(y)
            probs = out.predictive.probs
            acc = eval_acc(probs, y).item()

            running_val_loss.append(loss.item())
            running_val_acc.append(acc)
    return np.mean(running_val_loss), np.mean(running_val_acc)


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    print(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.basic.seed)

    exp_name = cfg.basic.exp_name
    save_dir = f'{exp_name}/{cfg.net_type}/{cfg.dataset}/vbll/{str(cfg.posterior)}/'
    os.makedirs(save_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    epochs = cfg.basic.n_epochs

    trainset, valset, model_cfg, model_kwargs = prepare_model(cfg.dataset, cfg.net_type)
    train_loader, val_loader, test_loader = data.getDataloader(trainset, valset, cfg.posterior.valid_size,
                                                            cfg.basic.batch_size, cfg.basic.num_workers,
                                                            split_train=True)

    model_args = list()
    model = DiscClassification(model_cfg.base, cfg.posterior.reg_param / len(trainset), *model_args, **model_kwargs).to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    
    for epoch in range(epochs):
        train_obj, train_acc = train_epoch(model, optimizer, train_loader, device=device)
        test_loss, test_acc = eval_epoch(model, test_loader, device=device)
        print("Epoch", epoch, "train_obj", train_obj, "train_acc", train_acc, "test_loss", test_loss, "test_acc", test_acc)

    torch.save(model.state_dict(), f'{save_dir}/seed{cfg.basic.seed}_model.pt')


if __name__ == '__main__':
    main()
