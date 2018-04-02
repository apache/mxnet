#!/usr/bin/perl
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

use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Base;
use Getopt::Long qw(HelpMessage);
use Time::HiRes qw(time);
use PDL::IO::Pic;

my $batch_size = 64;
my $nz  = 100;
my $ngf = 64;
my $ndf = 64;
my $nepoch = 25;
my $lr =0.0002;
my $beta1 = 0.5;
my $nc = 3;
## change to my $ctx = mx->cpu(); if needed
my $ctx = mx->gpu();

my $train_data = gluon->data->DataLoader(
    gluon->data->vision->MNIST('./data', train=>1, transform => \&transformer),
    batch_size=>$batch_size, shuffle=>1, last_batch=>'discard'
);

my $val_data = gluon->data->DataLoader(
    gluon->data->vision->MNIST('./data', train=>0, transform=> \&transformer),
    batch_size=>$batch_size, shuffle=>0
);

sub transformer
{
    my ($data, $label) = @_;
    # resize to 64x64
    $data = mx->image->imresize($data, 64, 64);
    $data = $data->reshape([1, 64, 64]);
    # normalize to [-1, 1]
    $data = $data->astype('float32')/128 - 1;
    # if image is greyscale, repeat 3 times to get RGB image.
    if($data->shape->[0] == 1)
    {
        $data = mx->nd->tile($data, [3, 1, 1]);
    }
    return ($data, $label);
}

sub visualize
{
    my ($data, $fake, $iter) = @_;
    mkdir "data_images";
    mkdir "data_images/$iter";
    mkdir "fake_images";
    mkdir "fake_images/$iter";
    for my $i (0..$batch_size-1)
    {
        my $d = ((pdl_shuffle($data->at($i)->at(0)->aspdl, [reverse(0..63)]) + 1)*128)->byte;
        my $f = ((pdl_shuffle($fake->at($i)->at(0)->aspdl, [reverse(0..63)]) + 1)*128)->byte;
        $d->wpic("data_images/$iter/$i.jpg");
        $f->wpic("fake_images/$iter/$i.jpg");
    }
}

# build the generator
my $netG = nn->Sequential();
$netG->name_scope(sub {
    # input is Z, going into a convolution
    $netG->add(nn->Conv2DTranspose($ngf * 8, 4, 1, 0, use_bias=>0));
    $netG->add(nn->BatchNorm());
    $netG->add(nn->Activation('relu'));
    # state size-> ($ngf*8) x 4 x 4
    $netG->add(nn->Conv2DTranspose($ngf * 4, 4, 2, 1, use_bias=>0));
    $netG->add(nn->BatchNorm());
    $netG->add(nn->Activation('relu'));
    # state size-> ($ngf*8) x 8 x 8
    $netG->add(nn->Conv2DTranspose($ngf * 2, 4, 2, 1, use_bias=>0));
    $netG->add(nn->BatchNorm());
    $netG->add(nn->Activation('relu'));
    # state size-> ($ngf*8) x 16 x 16
    $netG->add(nn->Conv2DTranspose($ngf, 4, 2, 1, use_bias=>0));
    $netG->add(nn->BatchNorm());
    $netG->add(nn->Activation('relu'));
    # state size-> ($ngf*8) x 32 x 32
    $netG->add(nn->Conv2DTranspose($nc, 4, 2, 1, use_bias=>0));
    $netG->add(nn->Activation('tanh'));
    # state size-> (nc) x 64 x 64
});

# build the discriminator
my $netD = nn->Sequential();
$netD->name_scope(sub {
    # input is (nc) x 64 x 64
    $netD->add(nn->Conv2D($ndf, 4, 2, 1, use_bias=>0));
    $netD->add(nn->LeakyReLU(0.2));
    # state size-> ($ndf) x 32 x 32
    $netD->add(nn->Conv2D($ndf * 2, 4, 2, 1, use_bias=>0));
    $netD->add(nn->BatchNorm());
    $netD->add(nn->LeakyReLU(0.2));
    # state size-> ($ndf) x 16 x 16
    $netD->add(nn->Conv2D($ndf * 4, 4, 2, 1, use_bias=>0));
    $netD->add(nn->BatchNorm());
    $netD->add(nn->LeakyReLU(0.2));
    # state size-> ($ndf) x 8 x 8
    $netD->add(nn->Conv2D($ndf * 8, 4, 2, 1, use_bias=>0));
    $netD->add(nn->BatchNorm());
    $netD->add(nn->LeakyReLU(0.2));
    # state size-> ($ndf) x 4 x 4
    $netD->add(nn->Conv2D(2, 4, 1, 0, use_bias=>0));
});

# loss
my $loss = gluon->loss->SoftmaxCrossEntropyLoss();

# initialize the generator and the discriminator
$netG->initialize(mx->init->Normal(0.02), ctx=>$ctx);
$netD->initialize(mx->init->Normal(0.02), ctx=>$ctx);

# trainer for the generator and the discriminator
my $trainerG = gluon->Trainer($netG->collect_params(), 'adam', {learning_rate => $lr, beta1 => $beta1});
my $trainerD = gluon->Trainer($netD->collect_params(), 'adam', {learning_rate => $lr, beta1 => $beta1});
# ============printing==============
my $real_label = mx->nd->ones([$batch_size], ctx=>$ctx);
my $fake_label = mx->nd->zeros([$batch_size], ctx=>$ctx);

my $metric = mx->metric->Accuracy();
print "Training...\n";

my $iter = 0;
for my $epoch (0..$nepoch-1)
{
    my $tic = time;
    my $btic = time;
    my $fake; my $data;
    while(defined(my $d = <$train_data>))
    {
        $data = $d->[0];
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        $data = $data->as_in_context($ctx);
        my $noise = mx->nd->random->normal(0, 1, shape=>[$batch_size, $nz, 1, 1], ctx=>$ctx);

        my ($output, $errD, $errG);
        autograd->record(sub {
            $output = $netD->($data);
            $output = $output->reshape([$batch_size, 2]);
            my $errD_real = $loss->($output, $real_label);
            $metric->update([$real_label], [$output]);

            $fake = $netG->($noise);
            $output = $netD->($fake->detach());
            $output = $output->reshape([$batch_size, 2]);
            my $errD_fake = $loss->($output, $fake_label);
            $errD = $errD_real + $errD_fake;
            $errD->backward();
            $metric->update([$fake_label], [$output]);
        });
        $trainerD->step($batch_size);

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        autograd->record(sub {
            $output = $netD->($fake);
            $output = $output->reshape([-1, 2]);
            $errG = $loss->($output, $real_label);
            $errG->backward();
        });

        $trainerG->step($batch_size);
        my ($name, $acc) = $metric->get();
        if(not $iter%100)
        {
            AI::MXNet::Logging->info("speed: %.2f samples/s", $batch_size / (time-$btic));
            AI::MXNet::Logging->info("discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d",
                mx->nd->mean($errD)->asscalar(), mx->nd->mean($errG)->asscalar(), $acc, $iter, $epoch);
        }
        $iter++;
        $btic = time;
    }
    my ($name, $acc) = $metric->get();
    $metric->reset();
    visualize($data, $fake, $epoch);
    AI::MXNet::Logging->info("\nbinary training acc at epoch %d: %s=%f", $epoch, $name, $acc);
    AI::MXNet::Logging->info("time: %f", time - $tic);
}
