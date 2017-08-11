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

### model
my $data = mx->symbol->Variable('data');
my $conv1= mx->symbol->Convolution(data => $data, name => 'conv1', num_filter => 32, kernel => [3,3], stride => [2,2]);
my $bn1  = mx->symbol->BatchNorm(data => $conv1, name => "bn1");
my $act1 = mx->symbol->Activation(data => $bn1, name => 'relu1', act_type => "relu");
my $mp1  = mx->symbol->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride =>[2,2], pool_type=>'max');

my $conv2= mx->symbol->Convolution(data => $mp1, name => 'conv2', num_filter => 32, kernel=>[3,3], stride=>[2,2]);
my $bn2  = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
my $act2 = mx->symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
my $mp2  = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');


my $fl   = mx->symbol->Flatten(data => $mp2, name=>"flatten");
my $fc1  = mx->symbol->FullyConnected(data => $fl,  name=>"fc1", num_hidden=>30);
my $act3 = mx->symbol->Activation(data => $fc1, name=>'relu3', act_type=>"relu");
my $fc2  = mx->symbol->FullyConnected(data => $act3, name=>'fc2', num_hidden=>10);
my $softmax = mx->symbol->SoftmaxOutput(data => $fc2, name => 'softmax');

## creates the image file in working directory, you need GraphViz installed for this to work
mx->viz->plot_network($softmax, save_format => 'png')->render("network.png");
