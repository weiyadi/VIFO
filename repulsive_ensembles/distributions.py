import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class Unorm_post():
    """
    Implementation of unnormalized posterior for a neural network model. It assume gaussian likelihood with variance
    config.pred_dist_std. The prior can be freely specified, the only requirement is a log_prob method to return the
    log probability of the particles.

    Args:
        ensemble: ensemble instance from MLP.py
        prior: prior instance from torch.distributions or custom, .log_prob() method is required
        config: Command-line arguments.
        n_train: number of training datapoints to rescale the likelihood

    """
    def __init__(self, ensemble, prior, n_train,add_prior = 'function'):
        self.prior = prior
        self.ensemble = ensemble
        self.num_train = n_train
        self.add_prior = add_prior


    def log_prob(self, particles, X, T, return_loss=False, return_pred = False, pred_idx = 1):
        pred = self.ensemble.forward(X, particles) # n_particles x n_batch x n_classes

        # if self.ensemble.net.classification:
            
        #     #pred = F.softmax(pred[1],2) #I have to do this to allow derivative and to not have nans 
        # else:
        #     # TODO: fix the regression case
        #     #loss = 0.5*torch.mean(F.mse_loss(prebpd[0], T, reduction='none'), 1)
        #     loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)

        loss = torch.stack([F.nll_loss(F.log_softmax(p, dim=1), T) for p in pred]) # n_particles, implicitly average over batch

        # ll = -loss*self.num_train / self.config.pred_dist_std ** 2

        if particles is None:
            particles = self.ensemble.particles

        if self.add_prior == 'weight':
            ll = -loss * self.num_train
            log_prob = torch.add(self.prior.log_prob(particles).sum(1), ll)
        elif self.add_prior == 'function':
            loss = -torch.add((self.prior.log_prob(pred).sum(2)).mean(1), -loss)
            log_prob = -loss * self.num_train
        else:
            ll = -loss * self.num_train
            log_prob = ll
#        log_prob = ll
        if return_loss:
            return torch.mean(loss),pred
        elif return_pred:
            return log_prob,pred #0 softmax, 1 is logit
        else:
            return log_prob