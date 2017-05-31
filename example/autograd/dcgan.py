import argparse
import mxnet as mx
from mxnet import nn
from mxnet.contrib import autograd
from data import cifar10_iterator


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
ctx = mx.gpu(0)

train_iter, val_iter = cifar10_iterator(opt.batchSize, (3, 64, 64), 64)


netG = nn.Sequential()
# input is Z, going into a convolution
netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, in_filters=nz, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 8))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 4 x 4
netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, in_filters=ngf * 8, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 4))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 8 x 8
netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, in_filters=ngf * 4, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf * 2))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 16 x 16
netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, in_filters=ngf * 2, use_bias=False))
netG.add(nn.BatchNorm(num_features=ngf))
netG.add(nn.Activation('relu'))
# state size. (ngf*8) x 32 x 32
netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, in_filters=ngf, use_bias=False))
netG.add(nn.Activation('tanh'))
# state size. (nc) x 64 x 64


netD = nn.Sequential()
# input is (nc) x 64 x 64
netD.add(nn.Conv2D(ndf, 4, 2, 1, in_filters=nc, use_bias=False))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 32 x 32
netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, in_filters=ndf, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 2))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 16 x 16
netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, in_filters=ndf * 2, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 4))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 8 x 8
netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, in_filters=ndf * 4, use_bias=False))
netD.add(nn.BatchNorm(num_features=ndf * 8))
netD.add(nn.LeakyReLU(0.2))
# state size. (ndf) x 4 x 4
netD.add(nn.Conv2D(2, 4, 1, 0, in_filters=ndf * 8, use_bias=False))
# netD.add(nn.Activation('sigmoid'))


netG.params.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.params.initialize(mx.init.Normal(0.02), ctx=ctx)


optimizerG = nn.Optim(netG.params, 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})
optimizerD = nn.Optim(netD.params, 'adam', {'learning_rate': opt.lr, 'beta1': opt.beta1})


real_label = mx.nd.ones((opt.batchSize,), ctx=ctx)
fake_label = mx.nd.zeros((opt.batchSize,), ctx=ctx)

for epoch in range(opt.niter):
    for batch in train_iter:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].copyto(ctx)
        noise = mx.nd.random_normal(0, 1, shape=(opt.batchSize, nz, 1, 1), ctx=ctx)

        with autograd.train_section():
            output = netD(data)
            output = output.reshape((opt.batchSize, 2))
            errD_real = nn.loss.softmax_cross_entropy_loss(output, real_label)

            fake = netG(noise)
            output = netD(fake.detach())
            output = output.reshape((opt.batchSize, 2))
            errD_fake = nn.loss.softmax_cross_entropy_loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()

        optimizerD.step(opt.batchSize)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.train_section():
            output = netD(fake)
            output = output.reshape((opt.batchSize, 2))
            errG = nn.loss.softmax_cross_entropy_loss(output, real_label)
            errG.backward()

        optimizerG.step(opt.batchSize)

        print mx.nd.mean(errD).asscalar(), mx.nd.mean(errG).asscalar()
