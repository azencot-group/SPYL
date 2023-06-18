import json
import utils
import progressbar
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from model import CDSVAE

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--nEpoch', default=1000, type=int, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
parser.add_argument('--neptune', default=True, type=bool, help='activate neptune tracking')

parser.add_argument('--dataset', default='Sprite2', type=str, help='dataset to train')
parser.add_argument("--dataset_path", default='/home/azencot_group/datasets/SPRITES_ICML/datasetICML/')
parser.add_argument('--frames', default=8, type=int, help='number of frames, 8 for sprite, 15 for digits and MUGs')
parser.add_argument('--channels', default=3, type=int, help='number of channels in images')
parser.add_argument('--image_width', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--decoder', default='ConvT', type=str, help='Upsampling+Conv or Transpose Conv: Conv or ConvT')

parser.add_argument('--f_rnn_layers', default=1, type=int, help='number of layers (content lstm)')
parser.add_argument('--rnn_size', default=256, type=int, help='dimensionality of hidden layer')
parser.add_argument('--f_dim', default=256, type=int, help='dim of f')
parser.add_argument('--z_dim', default=32, type=int, help='dimensionality of z_t')
parser.add_argument('--g_dim', default=128, type=int,
                    help='dimensionality of encoder output vector and decoder input vector')

parser.add_argument('--note', default='LogNCELoss', type=str, help='appx note')
parser.add_argument('--weight_f', default=1, type=float, help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', default=1, type=float, help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_c_aug', default=0, type=float, help='weighting on content contrastive loss')
parser.add_argument('--weight_m_aug', default=0, type=float, help='weighting on motion contrastive loss')
parser.add_argument('--weight_rec', default=1, type=float, help='weighting on reconstruction loss')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--c_loss', default=0, type=float, help='warmup epochs for contrastive loss')
parser.add_argument('--c_floss', default=90, type=float, help='weighting on motion contrastive loss')
parser.add_argument('--cneg', type=int, default=256, help='permutation aug')
parser.add_argument('--type_gt', type=str, default='action', help='action, skin, top, pant, hair')
parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

arguments = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu

mse_loss = nn.MSELoss().cuda()


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
            # (f_mean * y.transpose(0, 1)).sum(dim=0).sum(dim=-1)
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xn, dim=-1)
            denominator = torch.exp(torch.sum(xi * xn, dim=-1) / sim_mat_denom / self.tau)
        else:
            denominator = torch.exp(torch.sum(xi * xn, dim=-1) / self.tau)

        denominator = denominator.sum(dim=0)
        # new loss
        loss = torch.mean(-torch.log(numerator / (numerator + denominator)))

        return loss, torch.mean(numerator), torch.mean((numerator + denominator))


#  sample and predict KL divergance calculation for choosing the negatives for the contrastive loss
def kl_contrast(v_mean, v_logvar, v_bneg_mean, v_bneg_logvar, args):
    # compute neg samples
    v_var = torch.exp(v_logvar)
    v_mean, v_logvar, v_var = v_mean[:, None], v_logvar[:, None], v_var[:, None]
    # [b, 1, f_dim] -> original f_mean, f_logvar, f_var. unsqueeze the mid-dimension to be
    #                  able to compute a list of kl values per sample in the batch
    #                  against the bank and get a list per sample (from which we pick)

    v_neg_var = torch.exp(v_bneg_logvar)  # [sz, f_dim] -> bank f_mean, f_logvar, f_var
    KLF = 0.5 * torch.sum(v_bneg_logvar - v_logvar +
                          ((v_var + torch.pow(v_mean - v_bneg_mean, 2)) / v_neg_var) - 1, dim=-1)
    KK = torch.argsort(KLF, dim=-1, descending=True)
    JJ = KK.shape[-1] // 3

    if args.neg_mode == 'soft':  # taking from the first third (soft negatives)
        return v_bneg_mean[KK[:, :args.cneg]]
    if args.neg_mode == 'semi':  # taking from the first third (soft negatives)
        return v_bneg_mean[KK[:, JJ:JJ + args.cneg]]
    else:
        return v_bneg_mean[KK[:, :JJ * 2][:, torch.randperm(JJ * 2)[:args.cneg]]]


# ----- training functions -----
def train(x, model, optimizer, contras_fn, args, mode="train"):
    if mode == "train":
        model.zero_grad()

    batch_size = x.size(0)

    s_mean, s_logvar, s, d_post_mean, d_post_logvar, d_post, d_prior_mean, d_prior_logvar, d_prior, rec_x = model(x)

    # ----- calculate reconstruction loss ----- #
    l_recon = F.mse_loss(rec_x, x, reduction='sum')

    # ----- calculate KL of f ----- #
    s_mean = s_mean.view((-1, s_mean.shape[-1]))  # [128, 256]
    s_logvar = s_logvar.view((-1, s_logvar.shape[-1]))  # [128, 256]
    kld_f = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))

    # ----- calculate KL of z ----- #
    z_post_var = torch.exp(d_post_logvar)  # [128, 8, 32]
    z_prior_var = torch.exp(d_prior_logvar)  # [128, 8, 32]
    kld_z = 0.5 * torch.sum(d_prior_logvar - d_post_logvar +
                            ((z_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / z_prior_var) - 1)

    # ----- calculate KL of f ----- #
    l_recon, kld_f, kld_z = l_recon / batch_size, kld_f / batch_size, kld_z / batch_size
    batch_size, n_frame, z_dim = d_post_mean.size()

    # ----- calculate MI(x,f) and MI(x,z) ----- #

    # calculate the mutual infomation of f and z
    mi_sd = torch.zeros((1)).cuda()
    mi_sd = utils.calculate_mws(batch_size, d_post, d_post_logvar, d_post_mean, mi_sd, n_frame, args, s, s_logvar,
                                s_mean,
                                z_dim)

    loss = l_recon * args.weight_rec + kld_f * args.weight_f + kld_z * args.weight_z + mi_sd

    # ----- calculate MI(x,f) and MI(x,z) ----- #
    con_est_mi_s, con_est_mi_d = 0, 0

    # batch of random samples.
    with torch.no_grad():
        f_bneg_mean, f_bneg_logvar, z_bneg_mean, z_bneg_logvar = model.samples_tr(
            sz=(args.batch_size * 4))  # b for bank of potential neg

    if args.weight_c_aug != 0:
        c_aug = model.forward_fixed_content_for_classification_tr(x)
        f_mean_c, f_logvar_c, f_c, _, _, _, _, _, _, recon_c_aug = model(c_aug.cuda())

        # compute neg samples
        f_neg_mean = kl_contrast(s_mean, s_logvar, f_bneg_mean, f_bneg_logvar, args)

        con_est_mi_s, numerator_c, denominator_c = contras_fn(s_mean.squeeze(), f_mean_c, f_neg_mean)

        loss += con_est_mi_s * args.weight_c_aug

    if args.weight_m_aug != 0:
        # forward sampling new content (motion_aug), kinda cyclic
        m_aug = model.forward_fixed_action_for_classification_tr(x)
        _, _, _, z_post_mean_m, z_post_logvar_m, z_post_m, _, _, _, recon_m_aug = model(m_aug.cuda())

        d_post_mean, d_post_logvar, z_post_mean_m = torch.mean(d_post_mean, dim=1), torch.mean(d_post_logvar,
                                                                                               dim=1), torch.mean(
            z_post_mean_m, dim=1)
        # compute neg samples
        z_neg_mean = kl_contrast(d_post_mean, d_post_logvar, z_bneg_mean, z_bneg_logvar, args)

        con_est_mi_d, numerator_m, denominator_m = contras_fn(d_post_mean.view(batch_size, -1),
                                                              z_post_mean_m.view(batch_size, -1),
                                                              z_neg_mean)
        loss += con_est_mi_d * args.weight_m_aug

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return [i.data.cpu().numpy() for i in [l_recon, kld_f, kld_z, con_est_mi_s, con_est_mi_d, mi_sd]]


def main(args):
    utils.print_log('Running parameters:')
    utils.print_log(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # ----- init training -----
    args.optimizer = optim.Adam
    cdsvae = CDSVAE(args)
    cdsvae.apply(utils.init_weights)
    opt = args.optimizer(cdsvae.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #  transfer to gpu and parallelize if possible
    if torch.cuda.device_count() > 1:
        utils.print_log("Let's use {} GPUs!".format(torch.cuda.device_count()))
        cdsvae = nn.DataParallel(cdsvae)

    cdsvae = cdsvae.cuda()

    # ----- load a dataset -----
    train_data, test_data = utils.load_dataset(args)
    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=args.batch_size,  # 128
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=args.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    args.dataset_size = len(train_data)

    # ----- training loop -----
    for epoch in range(args.nEpoch):

        # sample and predict start after c_loss warmup epochs
        if epoch == args.c_loss:
            print('start contrastive loss computation')
            args.weight_c_aug, args.weight_m_aug = args.c_floss, args.c_floss

        cdsvae.train()
        epoch_loss.reset()

        args.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(maxval=len(train_loader)).start()

        # train loop
        for i, data in enumerate(train_loader):
            progress.update(i + 1)
            x, label_A, label_D = utils.reorder(data['images']), data['A_label'], data['D_label']
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            l_recon, kld_f, kld_z, con_est_mi_s, con_est_mi_d, mi_sd = train(x, cdsvae, opt, contras_fn, args)

            lr = opt.param_groups[0]['lr']
            epoch_loss.update(recon, kld_f, kld_z, con_est_mi_s, con_est_mi_d, mi_sd)

        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss.avg()
        utils.print_log('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_est_mi_s: %.5f |'
                        ' con_est_mi_d: %.5f | mi_sd: %.5f | lr: %.5f' % (
                            epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4], avg_loss[5], lr))

        # evaluation loop
        if epoch == args.nEpoch - 1 or epoch % args.evl_interval == 0:
            val_mse = val_kld_f = val_kld_z = val_con_est_mi_s = val_con_est_mi_d = val_mi_sd = 0.
            for i, data in enumerate(test_loader):
                x, label_A, label_D = utils.reorder(data['images']), data['A_label'], data['D_label']
                x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

                with torch.no_grad():
                    if epoch >= args.c_loss:
                        recon, kld_f, kld_z, con_est_mi_s, con_est_mi_d, mi_sd = train(x, label_A,
                                                                                       label_D, cdsvae,
                                                                                       opt, contras_fn,
                                                                                       args, mode="val")

                val_mse += recon
                val_kld_f += kld_f
                val_kld_z += kld_z
                val_con_est_mi_s += con_est_mi_s
                val_con_est_mi_d += con_est_mi_d
                val_mi_sd += mi_sd

            n_batch = len(test_loader)
            utils.print_log('[%02d] recon: %.2f | kld_f: %.2f | kld_z: %.2f | con_loss_c: %.5f |'
                            ' con_loss_m: %.5f | mi_sd: %.5f | lr: %.5f' % (
                                epoch, val_mse.item() / n_batch, val_kld_f.item() / n_batch, val_kld_z.item() / n_batch,
                                val_con_est_mi_s.item() / n_batch, val_con_est_mi_d.item() / n_batch,
                                val_mi_sd / n_batch,
                                lr))


#
if __name__ == '__main__':
    utils.init_seed(arguments.seed)

    # ----- load a dataset ----- #
    epoch_loss = utils.Loss()
    contras_fn = contrastive_loss(tau=0.5, normalize=True)

    main(arguments)
