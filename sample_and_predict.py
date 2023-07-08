from torch import nn as nn
import torch

from cdsvae.utils import reparameterize


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xp, xn):
        # xi: [b x dim]
        # xp: [b x dim] - means there is only one positive per sample, could be extended easily
        # xn: [b x cneg x dim] - cneg number of negatives per sample

        # transpose xn for computations
        xn = xn.transpose(0, 1)

        # top
        if self.normalize:  # False
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xp, dim=1)
            numerator = torch.exp(torch.sum(xi * xp, dim=-1) / sim_mat_denom / self.tau)
        else:
            numerator = torch.exp(torch.sum(xi * xp, dim=-1) / self.tau)

        # bottom
        if self.normalize:
            # (s_mean * y.transpose(0, 1)).sum(dim=0).sum(dim=-1)
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xn, dim=-1)
            denominator = torch.exp(torch.sum(xi * xn, dim=-1) / sim_mat_denom / self.tau)
        else:
            denominator = torch.exp(torch.sum(xi * xn, dim=-1) / self.tau)

        denominator = denominator.sum(dim=0)
        # new loss
        loss = torch.mean(-torch.log(numerator / (numerator + denominator)))

        return loss


#  sample and predict KL divergence calculation for choosing the negatives for the contrastive loss
def kl_contrast(v_mean, v_logvar, v_bneg_mean, v_bneg_logvar, args):
    # compute neg samples
    v_var = torch.exp(v_logvar)
    v_mean, v_logvar, v_var = v_mean[:, None], v_logvar[:, None], v_var[:, None]
    # [b, 1, s_dim] -> original s_mean, s_logvar, s_var. unsqueeze the mid-dimension to be
    #                  able to compute a list of kl values per sample in the batch
    #                  against the bank and get a list per sample (from which we pick)

    v_neg_var = torch.exp(v_bneg_logvar)  # [sz, s_dim] -> bank s_mean, s_logvar, s_var
    KLF = 0.5 * torch.sum(v_bneg_logvar - v_logvar +
                          ((v_var + torch.pow(v_mean - v_bneg_mean, 2)) / v_neg_var) - 1, dim=-1)
    KK = torch.argsort(KLF, dim=-1, descending=True)
    JJ = KK.shape[-1] // 3

    if args.neg_mode == 'soft':  # taking from the last third (soft negatives)
        return v_bneg_mean[KK[:, :args.cneg]]
    if args.neg_mode == 'semi':  # taking from the second third (medium negatives)
        return v_bneg_mean[KK[:, JJ:JJ + args.cneg]]
    else:
        # taking from the first third (hard negatives)
        return v_bneg_mean[KK[:, :JJ * 2][:, torch.randperm(JJ * 2)[:args.cneg]]]


def samples_tr(model, sz=96):
    # sz means the number of samples
    s_shape = (sz, model.s_dim)

    # sample f
    s_prior = reparameterize(torch.zeros(s_shape).cuda(), torch.zeros(s_shape).cuda(), random_sampling=True)
    s_expand = s_prior.unsqueeze(1).expand(-1, model.frames, model.s_dim)

    # sample z
    d_mean_prior, d_logvar_prior, d_out = model.sample_motion_prior(sz, model.frames, random_sampling=True)
    zf = torch.cat((d_mean_prior, s_expand), dim=2)
    recon_x_sample = model.decoder(zf)
    s_mean, s_logvar, s_post, d_mean_post, d_logvar_post, d_post = model.encode_and_sample_post(recon_x_sample)

    return s_mean, s_logvar, torch.mean(d_mean_post, dim=1), torch.mean(d_logvar_post, dim=1)
