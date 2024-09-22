# Variational Inference on the Final-Layer Output of Neural Networks
This repository includes the basic Python code for the following paper:

[Y. Wei and R. Khardon. Variational Inference on the Final-Layer Output of Neural Networks. TMLR.](https://openreview.net/forum?id=mTOzXLmLKr)

## Directory structure

This repository has the following directory structure
 * *README*: This file.
 * *data*: prepares data.
 * *models*: contains the implementation of the neural networks.
 * *posteriors*: contains the implementation of variational inference.
 * *config*: contains the configurations of different methods.
 * *helpers.py*: contains the helper functions to train and evaluate of the models.
 * *metrics.py*: contains the implementation of objectives and metrics.
 * *requirements.txt*: requirements of python packages.
 * *run_\<method>.py*: the script to run different methods, including dir, dropout, laplace, repulsive ensembles, swag, vbll, vi, vifo.
 * *uq_metric.py*: uncertainty quantification metrics.
 * *swag_utils.py*: functions only for swag.
 * *illustration.ipynb*: contains the example with artificial data.

## Run scripts
We adopt [Hydra](https://hydra.cc/) to apply the configurations to the script. 
Users can freely change the configurations, including the dataset, network, batch size, etc.
```console
python run_vifo.py +basic=image +posterior=vifo_mean +optimizer=adam +dataset=CIFAR10 +net_type=AlexNet
```

## Datasets and Networks
Datasets: CIFAR10, CIFAR100, SVHN, STL10.

Networks: AlexNet(Drop), PreResNet20(Drop). The suffix "Drop" must be included when running dropout.