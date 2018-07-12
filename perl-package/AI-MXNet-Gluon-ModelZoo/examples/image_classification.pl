#!/usr/bin/env perl
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
use AI::MXNet::Gluon::ModelZoo 'get_model';
use AI::MXNet::Gluon::Utils 'download';
use Getopt::Long qw(HelpMessage);

GetOptions(
    ## my Pembroke Welsh Corgi Kyuubi, enjoing Solar eclipse of August 21, 2017
    'image=s' => \(my $image = 'http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'.
                               'gluon/dataset/kyuubi.jpg'),
    'model=s' => \(my $model = 'resnet152_v2'),
    'help'    => sub { HelpMessage(0) },
) or HelpMessage(1);

## get a pretrained model (download parameters file if necessary)
my $net = get_model($model, pretrained => 1);

## ImageNet classes
my $fname = download('http://data.mxnet.io/models/imagenet/synset.txt');
my @text_labels = map { chomp; s/^\S+\s+//; $_ } IO::File->new($fname)->getlines;

## get the image from the disk or net
if($image =~ /^https/)
{
    eval { require IO::Socket::SSL; };
    die "Need to have IO::Socket::SSL installed for https images" if $@;
}
$image = $image =~ /^https?/ ? download($image) : $image;

# Following the conventional way of preprocessing ImageNet data:
# Resize the short edge into 256 pixes,
# And then perform a center crop to obtain a 224-by-224 image.
# The following code uses the image processing functions provided 
# in the AI::MXNet::Image module.

$image = mx->image->imread($image);
$image = mx->image->resize_short($image, $model =~ /inception/ ? 330 : 256);
($image) = mx->image->center_crop($image, [($model =~ /inception/ ? 299 : 224)x2]);

## CV that is used to read image is column major (as PDL)
$image = $image->transpose([2,0,1])->expand_dims(axis=>0);

## normalizing the image
my $rgb_mean = nd->array([0.485, 0.456, 0.406])->reshape([1,3,1,1]);
my $rgb_std = nd->array([0.229, 0.224, 0.225])->reshape([1,3,1,1]);
$image = ($image->astype('float32') / 255 - $rgb_mean) / $rgb_std;

# Now we can recognize the object in the image.
# We perform an additional softmax on the output to obtain probability scores.
# And then print the top-5 recognized objects.
my $prob = $net->($image)->softmax;
for my $idx (@{ $prob->topk(k=>5)->at(0) })
{
    my $i = $idx->asscalar;
    printf(
        "With prob = %.5f, it contains %s\n",
        $prob->at(0)->at($i)->asscalar, $text_labels[$i]
    );
}
