import torch
from torch.nn import functional as F

import metrics
import models
import numpy as np
import inspect

from metrics import outputs_to_log_probs_classfication, nll_regression
from metrics import logmeanexp

import data


def train_epoch_vi(vi_model, optimizer, criterion, trainloader, device=None, L=1, beta=0.1, regression=False,
                   scheduler=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vi_model(L, inputs)
        kl = vi_model.compute_reg(inputs)
        kl_list.append(kl.item())

        if not regression:
            loss = criterion(outputs, labels, kl, beta)
            loss.backward()

            log_outputs = F.log_softmax(outputs, dim=2)
            log_output = logmeanexp(log_outputs, dim=0)
            accs.append(metrics.acc(log_output.data, labels))
        else:
            loss = criterion(outputs, labels, kl, beta, regression=regression)
            loss.backward()

            accs.append(metrics.rmse(outputs, labels))
        training_loss += loss.cpu().data.numpy()
        nonan = True
        for name, param in vi_model.named_parameters():
            if param.requires_grad and torch.any(torch.isnan(param.grad)):
                nonan = False
                break
        if nonan:
            optimizer.step()
            if scheduler:
                scheduler.step()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def eval_epoch_vi(vi_model, testloader, device=None, L=1, regression=False):
    """Calculate ensemble accuracy and NLL Loss"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    test_loss = 0.0
    accs = []
    with torch.no_grad():
        for i, (input, labels) in enumerate(testloader):
            input, labels = input.to(device), labels.to(device)
            outputs = vi_model(L, input)
            if not regression:
                log_output = outputs_to_log_probs_classfication(outputs, aggregate='logmeanexp')
                test_loss += (F.nll_loss(log_output, labels, reduction='mean')).cpu().data.numpy()
                accs.append(metrics.acc(log_output.data, labels))
            else:
                nll = nll_regression(outputs, labels, aggregate='logmeanexp')
                test_loss += torch.mean(nll)
                accs.append(metrics.rmse(outputs, labels))
    return test_loss / len(testloader), np.mean(accs)


def train_epoch_standard(loader, model, criterion, optimizer, device=None, regression=False, regularization=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_loss = 0.0
    accs = []
    kl_list = []

    model.train()

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        loss, output, stats = criterion(model, inputs, labels)

        if 'kl' in stats.keys():
            kl_list.append(stats['kl'].cpu().data.numpy())

        if regularization is not None:
            reg_val = regularization(model)
            loss += reg_val
            kl_list.append(reg_val.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.data.item()

        if not regression:
            accs.append(metrics.acc(output.data, labels))
        else:
            accs.append(metrics.rmse(output, labels))

    return training_loss / len(loader), np.mean(accs), None if len(kl_list) == 0 else np.mean(kl_list)


def eval_epoch_standard(loader, model, criterion, device=None, regression=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loss = 0.0
    accs = []

    model.train()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            loss, output, stats = criterion(model, inputs, labels)
            test_loss += loss.item()

            if not regression:
                accs.append(metrics.acc(output, labels))
            else:
                accs.append(metrics.rmse(output, labels))

    return test_loss / len(loader), np.mean(accs)


def eval_epoch_swag(loader, swag_model, L=1, device=None, regression=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Perform Bayesian Model Averaging for SWAG model.
    test_loss = 0.0
    accs = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = []
            for _ in range(L):
                swag_model.sample(1.0)
                output = swag_model(inputs)
                outputs.append(output)
            if not regression:
                log_prob = outputs_to_log_probs_classfication(outputs, aggregate='logmeanexp')
                loss = F.nll_loss(log_prob, labels, reduction='mean').cpu().data.numpy()
                acc = metrics.acc(log_prob, labels)
            else:
                nll = nll_regression(outputs, labels, aggregate='logmeanexp')
                loss = torch.mean(nll, dim=0)
                acc = metrics.rmse(outputs, labels)
            test_loss += loss
            accs.append(acc)

    return test_loss / len(loader), np.mean(accs)


def train_epoch_vifo(vi_model, optimizer, criterion, trainloader, device=None, L=1, beta=0.1, regression=False,
                   scheduler=None, ood=None, beta_ood=None, ood_loader=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    kl_ood_list = []
    if ood_loader is not None:
        ood_iter = iter(ood_loader)
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, kl = vi_model(L, inputs, reg=True)
        kl_list.append(kl.item())

        if not regression:
            loss = criterion(outputs, labels, kl, beta)
            log_outputs = F.log_softmax(outputs, dim=2)
            log_output = logmeanexp(log_outputs, dim=0)
            accs.append(metrics.acc(log_output.data, labels))
        else:
            loss = criterion(outputs, labels, kl, beta, regression=regression)
            accs.append(metrics.rmse(outputs, labels))
        
        
        if ood is not None and beta_ood > 0:
            if ood_loader is not None:
                inputs_ood, _ = next(ood_iter)
                inputs_ood = inputs_ood.to(device)
            elif ood == 'uniform':
                d = torch.max(inputs) - torch.min(inputs)
                inputs_ood = torch.rand(inputs.shape).to(device) * (torch.max(inputs) + d - torch.min(inputs)) + torch.min(inputs) - d / 2      
            else:
                raise NotImplementedError
            _, kl_ood = vi_model(L, inputs_ood, reg=True)
            loss += beta_ood * kl_ood
            kl_ood_list.append(kl_ood.item())
        loss.backward()
        
        training_loss += loss.cpu().data.numpy()
        nonan = True
        for name, param in vi_model.named_parameters():
            if param.requires_grad and torch.any(torch.isnan(param.grad)):
                nonan = False
                break
        if nonan:
            optimizer.step()
            if scheduler:
                scheduler.step()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list), np.mean(kl_ood_list) if len(kl_ood_list) > 0 else None


def prepare_model(dataset, net_type, **kwargs):
    model_cfg = getattr(models, net_type)
    trainset, valset, inputs, outputs = data.getTransformedDataset(dataset, model_cfg, **kwargs)
    num_classes = outputs
    regression = kwargs.get('uci_regression', False)

    model_kwargs = model_cfg.kwargs
    if 'in_channels' in inspect.getfullargspec(model_cfg.base.__init__).args:
        if dataset == 'MNIST':
            model_kwargs['in_channels'] = 1
        else:
            model_kwargs['in_channels'] = 3
    if not regression:
        model_kwargs['num_classes'] = num_classes
        if 'mlp' in str(model_cfg) and 'in_dim' not in kwargs:
            model_kwargs['in_dim'] = inputs
    else:
        model_kwargs['in_dim'] = inputs
    return trainset, valset, model_cfg, model_kwargs
