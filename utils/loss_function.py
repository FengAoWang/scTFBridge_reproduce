import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def KL_loss(mu, logvar, beta):
    # KL divergence loss
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return beta * KLD.cuda()


def GEOAE_loss(mu, log_var, beta):
    KLD = torch.linalg.norm(log_var.exp() - torch.ones_like(log_var), dim=1).sum()
    return beta * KLD


def Reconstruction_loss(recon_x, x, recon_param, dist):
    batch_size = x.size(0)
    if dist == 'bernoulli':
        BCE = nn.BCEWithLogitsLoss(reduction='sum')
        recons_loss = BCE(recon_x, x) / batch_size
    elif dist == 'gaussian':
        mse = nn.MSELoss(reduction='sum')
        recons_loss = mse(recon_x, x)
    elif dist == 'F2norm':
        recons_loss = torch.norm(recon_x-x, p=2)
    elif dist == 'prob':
        recons_loss = recon_x.log_prob(x).sum(dim=1).mean(dim=0)
    elif dist == 'Poisson':
        pois_loss = nn.PoissonNLLLoss(full=True, reduction='sum')
        recons_loss = pois_loss(recon_x, x)
    elif dist == 'ce':
        x = torch.argmax(x, dim=1)
        ce_loss = nn.CrossEntropyLoss(reduction='sum')
        recons_loss = ce_loss(recon_x, x)
    else:
        raise AttributeError("invalid dist")

    return recon_param * recons_loss


def cox_loss(survtime, censor, hazard_pred):
    current_batch_len = len(survtime)

    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox




