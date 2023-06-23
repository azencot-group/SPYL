import json

import numpy as np

import utils
import cdsvae.utils as cdsvae_utils
import progressbar
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from cdsvae.model import CDSVAE, classifier_Sprite_all
from cdsvae.utils import kl_loss_calc, KL_divergence, inception_score, entropy_Hy, entropy_Hyx
from sample_and_predict import contrastive_loss, kl_contrast, samples_tr


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2.e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--nEpoch', default=1000, type=int, help='number of epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--evl_interval', default=10, type=int, help='evaluate every n epoch')
    parser.add_argument('--log_dir', default='./logs', type=str, help='base directory to save logs')
    parser.add_argument('--dataset', default='Sprite2', type=str, help='dataset to train')
    parser.add_argument("--dataset_path",
                        default='/home/azencot_group/datasets/SPRITES_ICML/datasetICML/')  # TODO change to empty
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
    parser.add_argument('--weight_s', default=1, type=float, help='weighting on KL to prior, content vector')
    parser.add_argument('--weight_d', default=1, type=float, help='weighting on KL to prior, motion vector')
    parser.add_argument('--weight_rec', default=1, type=float, help='weighting on reconstruction loss')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--type_gt', type=str, default='action',
                        help='action, skin, top, pant, hair')
    parser.add_argument('--niter', type=int, default=5, help='number of runs for testing')

    # ----- sample and predict arguments ----- #
    """ the below two arguments starts with 0 and changes upon the end of c_loss warmup. gets the value of c_floss"""
    parser.add_argument('--weight_c_aug', default=0, type=float, help='weighting on content contrastive loss')
    parser.add_argument('--weight_m_aug', default=0, type=float, help='weighting on motion contrastive loss')

    parser.add_argument('--c_loss', default=50, type=float, help='warmup epochs for contrastive loss')
    parser.add_argument('--c_floss', default=90, type=float, help='weighting on motion contrastive loss')

    parser.add_argument('--neg_mode', type=str, default='soft',
                        help='The third which we going to sample negatives from.'
                             ' The options are: soft, semi, hard.'
                             ' soft are from the last third '
                             '(the farset from the datapoint)')
    parser.add_argument('--cneg', type=int, default=256, help='the number of negative samples to choose')

    return parser


parser = arg_parse()

arguments = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = arguments.gpu

mse_loss = nn.MSELoss().cuda()


# ----- evaluation functions -----
def check_cls(cdsvae, classifier, test_loader, args):
    e_values_action, e_values_skin, e_values_pant, e_values_top, e_values_hair = [], [], [], [], []
    for epoch in range(args.niter):

        print("Epoch", epoch)
        cdsvae.eval()
        mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4 = 0, 0, 0, 0, 0
        mean_acc0_sample, mean_acc1_sample, mean_acc2_sample, mean_acc3_sample, mean_acc4_sample = 0, 0, 0, 0, 0
        pred1_all, pred2_all, label2_all = list(), list(), list()
        label_gt = list()
        for i, data in enumerate(test_loader):
            x, label_A, label_D = utils.reorder(data['images']), data['A_label'][:, 0], data['D_label'][:, 0]
            x, label_A, label_D = x.cuda(), label_A.cuda(), label_D.cuda()

            if args.type_gt == "action":
                recon_x_sample, recon_x = cdsvae.forward_fixed_action_for_classification(x)
            else:
                recon_x_sample, recon_x = cdsvae.forward_fixed_content_for_classification(x)

            with torch.no_grad():
                pred_action1, pred_skin1, pred_pant1, pred_top1, pred_hair1 = classifier(x)
                pred_action2, pred_skin2, pred_pant2, pred_top2, pred_hair2 = classifier(recon_x_sample)
                pred_action3, pred_skin3, pred_pant3, pred_top3, pred_hair3 = classifier(recon_x)

                pred1 = F.softmax(pred_action1, dim=1)
                pred2 = F.softmax(pred_action2, dim=1)
                pred3 = F.softmax(pred_action3, dim=1)

            label1 = np.argmax(pred1.detach().cpu().numpy(), axis=1)
            label2 = np.argmax(pred2.detach().cpu().numpy(), axis=1)
            label3 = np.argmax(pred3.detach().cpu().numpy(), axis=1)
            label2_all.append(label2)

            pred1_all.append(pred1.detach().cpu().numpy())
            pred2_all.append(pred2.detach().cpu().numpy())
            label_gt.append(np.argmax(label_D.detach().cpu().numpy(), axis=1))

            def count_D(pred, label, mode=1):
                return (pred // mode) == (label // mode)

            # action
            acc0_sample = (np.argmax(pred_action2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_D.cpu().numpy(), axis=1)).mean()
            # skin
            acc1_sample = (np.argmax(pred_skin2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 0].cpu().numpy(), axis=1)).mean()
            # pant
            acc2_sample = (np.argmax(pred_pant2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 1].cpu().numpy(), axis=1)).mean()
            # top
            acc3_sample = (np.argmax(pred_top2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 2].cpu().numpy(), axis=1)).mean()
            # hair
            acc4_sample = (np.argmax(pred_hair2.detach().cpu().numpy(), axis=1)
                           == np.argmax(label_A[:, 3].cpu().numpy(), axis=1)).mean()
            mean_acc0_sample += acc0_sample
            mean_acc1_sample += acc1_sample
            mean_acc2_sample += acc2_sample
            mean_acc3_sample += acc3_sample
            mean_acc4_sample += acc4_sample

        print(
            'Test sample: action_Acc: {:.2f}% skin_Acc: {:.2f}% pant_Acc: {:.2f}% top_Acc: {:.2f}% hair_Acc: {:.2f}% '.format(
                mean_acc0_sample / len(test_loader) * 100,
                mean_acc1_sample / len(test_loader) * 100, mean_acc2_sample / len(test_loader) * 100,
                mean_acc3_sample / len(test_loader) * 100, mean_acc4_sample / len(test_loader) * 100))

        label2_all = np.hstack(label2_all)
        label_gt = np.hstack(label_gt)
        pred1_all = np.vstack(pred1_all)
        pred2_all = np.vstack(pred2_all)

        acc = (label_gt == label2_all).mean()
        kl = KL_divergence(pred2_all, pred1_all)

        nSample_per_cls = min([(label_gt == i).sum() for i in np.unique(label_gt)])
        index = np.hstack([np.nonzero(label_gt == i)[0][:nSample_per_cls] for i in np.unique(label_gt)]).squeeze()
        pred2_selected = pred2_all[index]

        IS = inception_score(pred2_selected)
        H_yx = entropy_Hyx(pred2_selected)
        H_y = entropy_Hy(pred2_selected)

        print('acc: {:.2f}%, kl: {:.4f}, IS: {:.4f}, H_yx: {:.4f}, H_y: {:.4f}'.format(acc * 100, kl, IS, H_yx, H_y))

        e_values_action.append(mean_acc0_sample / len(test_loader) * 100)
        e_values_skin.append(mean_acc1_sample / len(test_loader) * 100)
        e_values_pant.append(mean_acc2_sample / len(test_loader) * 100)
        e_values_top.append(mean_acc3_sample / len(test_loader) * 100)
        e_values_hair.append(mean_acc4_sample / len(test_loader) * 100)

    print(
        'final | acc: {:.2f}% | acc: {:.2f}% |acc: {:.2f}% |acc: {:.2f}% |acc: {:.2f}%'.format(np.mean(e_values_action),
                                                                                               np.mean(e_values_skin),
                                                                                               np.mean(e_values_pant),
                                                                                               np.mean(e_values_top),
                                                                                               np.mean(e_values_hair)))


def main(args):
    utils.print_log('Running parameters:')
    utils.print_log(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # ----- init training -----
    args.optimizer = optim.Adam
    cdsvae = CDSVAE(args)
    # cdsvae.load_state_dict() # TODO

    #  transfer to gpu and parallelize if possible
    if torch.cuda.device_count() > 1:
        cdsvae = nn.DataParallel(cdsvae)

    cdsvae = cdsvae.cuda()

    # ----- load a dataset -----
    _, test_data = utils.load_dataset(args)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=args.batch_size,  # 128
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    # ----- quantitative evaluation of swap generations -----
    check_cls(cdsvae, classifier, test_loader, args)

    # ----- qualitative swap example -----
    batch = next(iter(test_loader))
    swapped_batch = cdsvae.forward_exchange(batch)

    x1 = batch[0]
    x2 = batch[1]
    x1_s_x2_d = swapped_batch[0]
    x1_d_x2_s = swapped_batch[1]

    utils.imshow_seqeunce(x1, title="x1")
    utils.imshow_seqeunce(x2, title="x2")
    utils.imshow_seqeunce(x1_s_x2_d, title="x1 statics x2 dynamics")
    utils.imshow_seqeunce(x1_d_x2_s, title="x1 dynamics x2 statics")

#
if __name__ == '__main__':
    utils.init_seed(arguments.seed)

    # ----- load a dataset ----- #
    epoch_loss = utils.Loss()
    contras_fn = contrastive_loss(tau=0.5, normalize=True)

    # ----- load classifier ----- #
    classifier = classifier_Sprite_all(arguments)
    arguments.resume = 'cdsvae/sprite_judge.tar'
    loaded_dict = torch.load(arguments.resume)
    classifier.load_state_dict(loaded_dict['state_dict'])
    classifier = classifier.cuda().eval()

    main(arguments)

# TODO swap experiment ?


# TODO - batch metric evaluation ?
