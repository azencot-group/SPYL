import numpy as np
import torch
import torch as nn


# ---------------- encoder -----------------------
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


# ---------------- decoder -----------------------
"""
# Using transpose conv as the block to up-sample
"""


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class decoder_convT(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_convT, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


# ---------------- model -----------------------
def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


class CDSVAE(nn.Module):
    def __init__(self, opt):
        super(CDSVAE, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim  # frame/image feature
        self.channels = opt.channels  # image channel
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames

        self.encoder = encoder(self.g_dim, self.channels)
        self.decoder = decoder_convT(self.z_dim + self.f_dim, self.channels)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # ----- Posterior of content and motion
        # content and motion features share one bi-lstm
        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim * 2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim * 2, self.f_dim)

        # motion features from the next RNN
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        # pass the bidirectional lstm
        lstm_out, _ = self.z_lstm(conv_x)
        # get f:
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = reparameterize(f_mean, f_logvar, random_sampling=True)

        # pass to one direction rnn
        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = reparameterize(z_mean, z_logvar, random_sampling=True)

        if isinstance(x, list):
            f_mean_list = [f_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.z_lstm(conv_x)
                # get f:
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_mean_list.append(f_mean)
            f_mean = f_mean_list
        # f_mean is list if triple else not
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    # ------ sample z from learned LSTM prior base on previous postior, teacher forcing for training  ------
    def sample_motion_prior_train(self, z_post, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:, i, :]
        return z_means, z_logvars, z_out

    # ------ sample z purely from learned LSTM prior with arbitrary frames------
    def sample_motion_prior(self, n_sample, n_frame, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            # two layer LSTM and two one-layer FC
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_motion_prior_train(z_post, random_sampling=True)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x

    def encoder_frame(self, x):
        # input x is list of length Frames [batchsize, channels, size, size]
        # convert it to [batchsize, frames, channels, size, size]
        # x = torch.stack(x, dim=1)
        # [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        # to [batch_size , frames, embed_dim]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    # fixed content and sample motion for classification disagreement scores
    def forward_fixed_content_for_classification(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                 random_sampling=True)
        # f_prior = reparameterize(f_mean, torch.zeros(f_logvar.shape).cuda(), random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        # zf = torch.cat((z_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_fixed_content_for_classification_tr(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(x.size(0), self.frames, random_sampling=True)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    # sample content and fixed motion for classification disagreement scores
    def forward_fixed_action_for_classification_tr(self, x):
        # z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        # f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        f_prior = reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                 random_sampling=True)
        # f_prior = reparameterize(f_mean, torch.zeros(f_logvar.shape).cuda(), random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        # zf = torch.cat((z_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        return recon_x_sample

    def samples_tr(self, sz=96):
        # sz means the number of samples
        f_shape = (sz, self.f_dim)

        # sample f
        f_prior = reparameterize(torch.zeros(f_shape).cuda(), torch.zeros(f_shape).cuda(), random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        # sample z
        z_mean_prior, z_logvar_prior, z_out = self.sample_motion_prior(sz, self.frames, random_sampling=True)
        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(recon_x_sample)

        return f_mean, f_logvar, torch.mean(z_mean_post, dim=1), torch.mean(z_logvar_post, dim=1)

    def forward_exchange(self, x):
        f_mean, f_logvar, f, z_mean_post, z_logvar_post, z = self.encode_and_sample_post(x)

        # perm = torch.LongTensor(np.random.permutation(f.shape[0]))
        # f_mix = f[perm]

        a = f[np.arange(0, f.shape[0], 2)]
        b = f[np.arange(1, f.shape[0], 2)]
        f_mix = torch.stack((b, a), dim=1).view((-1, f.shape[1]))
        # mix = torch.stack((b[0], a[0], b[1], a[1], b[2], a[2], b[3], a[3], b[4], a[4]), dim=0)
        # f_mix = torch.cat((mix, a[5:], b[5:]))

        f_expand = f_mix.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f, None, None, z, None, None, recon_x
