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
use AI::MXNet::Gluon::Utils qw(download);
use AI::MXNet 'mx';
use AI::MXNet::Gluon::NN 'nn';
use PDL::IO::Pic;
require './net.pl';
require './utils.pl';
use Getopt::Long qw(HelpMessage);

GetOptions(
    'content-image=s' => \(my $content_image),
    'style-image=s'   => \(my $style_image),
    'model=s'         => \(my $model = './data/msgnet_21styles-2cb88353.params'),
    'output-image=s'  => \(my $output_image = 'out.jpg'),
    'content-size=i'  => \(my $content_size = 512),
    'ngf'             => \(my $ngf = 128), ## number of convolutional filters for the model
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

die "Please supply --content-image <path or url> and --style-image <path or url>"
    unless (defined $content_image and defined $style_image);
if($content_image =~ /^https:/ or $style_image =~ /^https:/)
{
    eval { require IO::Socket::SSL; };
    die "You need to have IO::Socket::SSL installed for https images" if $@;
}
$content_image = download($content_image) if $content_image =~ /^https?:/;
$style_image = download($style_image) if $style_image =~ /^https?:/;

evaluate(
    content_image  => $content_image,
    style_image    => $style_image,
    content_size   => $content_size,
    style_size     => $content_size,
    output_image   => $output_image,
    ngf            => $ngf,
    model          => $model
);
