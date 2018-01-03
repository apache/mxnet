# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
from PIL import Image

from mxnet import autograd, gluon
from mxnet.gluon import nn, Block, HybridBlock, Parameter, ParameterDict
import mxnet.ndarray as F

import net
import utils
from option import Options
import data

def train(args):
    np.random.seed(args.seed)
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # dataloader
    transform = utils.Compose([utils.Scale(args.image_size),
                               utils.CenterCrop(args.image_size),
                               utils.ToTensor(ctx),
                               ])
    train_dataset = data.ImageFolder(args.dataset, transform)
    train_loader = gluon.data.DataLoader(train_dataset, batch_size=args.batch_size, last_batch='discard')
    style_loader = utils.StyleLoader(args.style_folder, args.style_size, ctx=ctx)
    print('len(style_loader):',style_loader.size())
    # models
    vgg = net.Vgg16()
    utils.init_vgg_params(vgg, 'models', ctx=ctx)
    style_model = net.Net(ngf=args.ngf)
    style_model.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        style_model.collect_params().load(args.resume, ctx=ctx)
    print('style_model:',style_model)
    # optimizer and loss
    trainer = gluon.Trainer(style_model.collect_params(), 'adam',
                            {'learning_rate': args.lr})
    mse_loss = gluon.loss.L2Loss()

    for e in range(args.epochs):
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            # prepare data
            style_image = style_loader.get(batch_id)
            style_v = utils.subtract_imagenet_mean_preprocess_batch(style_image.copy())
            style_image = utils.preprocess_batch(style_image)

            features_style = vgg(style_v)
            gram_style = [net.gram_matrix(y) for y in features_style]

            xc = utils.subtract_imagenet_mean_preprocess_batch(x.copy())
            f_xc_c = vgg(xc)[1]
            with autograd.record():
                style_model.setTarget(style_image)
                y = style_model(x)

                y = utils.subtract_imagenet_mean_batch(y)
                features_y = vgg(y)

                content_loss = 2 * args.content_weight * mse_loss(features_y[1], f_xc_c)

                style_loss = 0.
                for m in range(len(features_y)):
                    gram_y = net.gram_matrix(features_y[m])
                    _, C, _ = gram_style[m].shape
                    gram_s = F.expand_dims(gram_style[m], 0).broadcast_to((args.batch_size, 1, C, C))
                    style_loss = style_loss + 2 * args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

                total_loss = content_loss + style_loss
                total_loss.backward()
                
            trainer.step(args.batch_size)
            mx.nd.waitall()

            agg_content_loss += content_loss[0]
            agg_style_loss += style_loss[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss.asnumpy()[0] / (batch_id + 1),
                                agg_style_loss.asnumpy()[0] / (batch_id + 1),
                                (agg_content_loss + agg_style_loss).asnumpy()[0] / (batch_id + 1)
                )
                print(mesg)

            
            if (batch_id + 1) % (4 * args.log_interval) == 0:
                # save model
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                    args.content_weight) + "_" + str(args.style_weight) + ".params"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                style_model.collect_params().save(save_model_path)
                print("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".params"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    style_model.collect_params().save(save_model_path)
    print("\nDone, trained model saved at", save_model_path)


def evaluate(args):
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # images
    content_image = utils.tensor_load_rgbimage(args.content_image,ctx, size=args.content_size, keep_asp=True)
    style_image = utils.tensor_load_rgbimage(args.style_image, ctx, size=args.style_size)
    style_image = utils.preprocess_batch(style_image)
    # model
    style_model = net.Net(ngf=args.ngf)
    style_model.collect_params().load(args.model, ctx=ctx)
    # forward
    style_model.setTarget(style_image)
    output = style_model(content_image)
    utils.tensor_save_bgrimage(output[0], args.output_image, args.cuda)


def optimize(args):
    """    Gatys et al. CVPR 2017
    ref: Image Style Transfer Using Convolutional Neural Networks
    """
    if args.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    # load the content and style target
    content_image = utils.tensor_load_rgbimage(args.content_image,ctx, size=args.content_size, keep_asp=True)
    content_image = utils.subtract_imagenet_mean_preprocess_batch(content_image)
    style_image = utils.tensor_load_rgbimage(args.style_image, ctx, size=args.style_size)
    style_image = utils.subtract_imagenet_mean_preprocess_batch(style_image)
    # load the pre-trained vgg-16 and extract features
    vgg = net.Vgg16()
    utils.init_vgg_params(vgg, 'models', ctx=ctx)
    # content feature
    f_xc_c = vgg(content_image)[1]
    # style feature
    features_style = vgg(style_image)
    gram_style = [net.gram_matrix(y) for y in features_style]
    # output
    output = Parameter('output', shape=content_image.shape)
    output.initialize(ctx=ctx)
    output.set_data(content_image)
    # optimizer
    trainer = gluon.Trainer([output], 'adam',
                            {'learning_rate': args.lr})
    mse_loss = gluon.loss.L2Loss()

    # optimizing the images
    for e in range(args.iters):
        utils.imagenet_clamp_batch(output.data(), 0, 255)
        # fix BN for pre-trained vgg
        with autograd.record():
            features_y = vgg(output.data())
            content_loss = 2 * args.content_weight * mse_loss(features_y[1], f_xc_c)
            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = net.gram_matrix(features_y[m])
                gram_s = gram_style[m]
                style_loss = style_loss + 2 * args.style_weight * mse_loss(gram_y, gram_s)
            total_loss = content_loss + style_loss
            total_loss.backward()

        trainer.step(1)
        if (e + 1) % args.log_interval == 0:
            print('loss:{:.2f}'.format(total_loss.asnumpy()[0]))
        
    # save the image
    output = utils.add_imagenet_mean_batch(output.data())
    utils.tensor_save_bgrimage(output[0], args.output_image, args.cuda)


def main():
    # figure out the experiments type
    args = Options().parse()

    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")

    if args.subcommand == "train":
        # Training the model 
        train(args)

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    elif args.subcommand == 'optim':
        # Gatys et al. using optimization-based approach
        optimize(args)

    else:
        raise ValueError('Unknow experiment type')


if __name__ == "__main__":
   main()
