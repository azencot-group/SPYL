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


# -------------------------- monitoring utils -------------------------- #
def t_to_np(x):
    return x.detach().cpu().numpy()


def imshow_seqeunce(DATA, plot=True, titles=None, figsize=(50, 10), fontsize=50, title=""):
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
    plt.title(title)

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
