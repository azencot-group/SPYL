import os, sys
import torch.nn.functional as F
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch as nn
import numpy as np
from dataloader.sprite import Sprite
import matplotlib.pyplot as plt
import random


# -------------------------- general utils -------------------------- #
def init_seed(seed):
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def print_log(print_string, log=None, verbose=True):
    if verbose:
        print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()


# -------------------------- model utils -------------------------- #

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def load_checkpoint(model, checkpoint_name):
    print("Loading Checkpoint from '{}'".format(checkpoint_name))
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])


# -------------------------- monitoring utils -------------------------- #
def t_to_np(x):
    return x.detach().cpu().numpy()


def imshow_seqeunce(DATA, plot=True, titles=None, figsize=(50, 10), fontsize=50):
    rc = 2 * len(DATA[0])
    fig, axs = plt.subplots(rc, 2, figsize=figsize)

    for ii, data in enumerate(DATA):
        for jj, img in enumerate(data):

            img = t_to_np(img)
            tsz, csz, hsz, wsz = img.shape
            img = img.transpose((2, 0, 3, 1)).reshape((hsz, tsz * wsz, -1))

            ri, ci = jj * 2 + ii // 2, ii % 2
            axs[ri][ci].imshow(img)
            axs[ri][ci].set_axis_off()
            if titles is not None:
                axs[ri][ci].set_title(titles[ii][jj], fontsize=fontsize)

    plt.subplots_adjust(wspace=.05, hspace=0)

    if plot:
        plt.show()


class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, recon, kld_f, kld_z, con_est_mi_s, con_est_mi_d, mi_sd):
        self.recon.append(recon)
        self.kld_f.append(kld_f)
        self.kld_z.append(kld_z)
        self.con_est_mi_s.append(con_est_mi_s)
        self.con_est_mi_d.append(con_est_mi_d)
        self.mi_sd.append(mi_sd)

    def reset(self):
        self.recon = []
        self.kld_f = []
        self.kld_z = []
        self.con_est_mi_s = []
        self.con_est_mi_d = []
        self.mi_sd = []

    def avg(self):
        return [np.asarray(i).mean() for i in
                [self.recon, self.kld_f, self.kld_z, self.con_est_mi_s, self.con_est_mi_d, self.mi_sd]]


# -------------------------- data utils -------------------------- #

# X, X, 64, 64, 3 -> # X, X, 3, 64, 64
def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)


def load_dataset(args):
    path = args.dataset_path
    with open(path + 'sprites_X_train.npy', 'rb') as f:
        X_train = np.load(f)
    with open(path + 'sprites_X_test.npy', 'rb') as f:
        X_test = np.load(f)
    with open(path + 'sprites_A_train.npy', 'rb') as f:
        A_train = np.load(f)
    with open(path + 'sprites_A_test.npy', 'rb') as f:
        A_test = np.load(f)
    with open(path + 'sprites_D_train.npy', 'rb') as f:
        D_train = np.load(f)
    with open(path + 'sprites_D_test.npy', 'rb') as f:
        D_test = np.load(f)

    train_data = Sprite(data=X_train, A_label=A_train, D_label=D_train)
    test_data = Sprite(data=X_test, A_label=A_test, D_label=D_test)

    return train_data, test_data


# -------------------------- computational utils -------------------------- #

def matrix_log_density_gaussian(x, mu, logvar):
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu) ** 2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def compute_mi(latent_sample, latent_dist):
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                         latent_dist,
                                                                         None,
                                                                         is_mss=False)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()

    return mi_loss


def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    # print("latent_sample:", latent_sample.shape)
    # print("latent_dist:", len(latent_dist), latent_dist[0].shape, latent_dist[1].shape)
    # print("is_mss:", is_mss)

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    # zeros = torch.zeros_like(latent_sample)
    # log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    # log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    # return log_pz, log_qz, log_prod_qzi, log_q_zCx
    return None, log_qz, None, log_q_zCx


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')


def log_density(sample, mu, logsigma):
    mu = mu.type_as(sample)
    logsigma = logsigma.type_as(sample)
    c = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)

    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logsigma + c)


def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def calculate_mws(batch_size, d_post, d_post_logvar, d_post_mean, mi_fz, n_frame, opt, s, s_logvar, s_mean, z_dim):
    # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
    # batch_size x batch_size x f_dim
    _logq_f_tmp = log_density(s.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, batch_size, 1, opt.f_dim),
                              # [8, 128, 1, 256]
                              s_mean.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size, opt.f_dim),
                              # [8, 1, 128, 256]
                              s_logvar.unsqueeze(0).repeat(n_frame, 1, 1).view(n_frame, 1, batch_size,
                                                                               opt.f_dim))  # [8, 1, 128, 256]
    # n_frame x batch_size x batch_size x f_dim
    _logq_z_tmp = log_density(d_post.transpose(0, 1).view(n_frame, batch_size, 1, z_dim),  # [8, 128, 1, 32]
                              d_post_mean.transpose(0, 1).view(n_frame, 1, batch_size, z_dim),  # [8, 1, 128, 32]
                              d_post_logvar.transpose(0, 1).view(n_frame, 1, batch_size, z_dim))  # [8, 1, 128, 32]
    _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)  # [8, 128, 128, 288]
    logq_f = (logsumexp(_logq_f_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    logq_z = (logsumexp(_logq_z_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    logq_fz = (logsumexp(_logq_fz_tmp.sum(3), dim=2, keepdim=False) - math.log(
        batch_size * opt.dataset_size))  # [8, 128]
    # n_frame x batch_size
    mi_fz = F.relu(logq_fz - logq_f - logq_z).mean()
    return mi_fz
