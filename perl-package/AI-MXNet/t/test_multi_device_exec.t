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
use Test::More tests => 10;
use AI::MXNet qw(mx);
use AI::MXNet::Base;

sub test_ctx_group
{
    my ($data, $fc1, $act1);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage1');
        $data = mx->symbol->Variable('data');
        $fc1  = mx->symbol->FullyConnected(data => $data, name=>'fc1', num_hidden=>128);
        $act1 = mx->symbol->Activation(data => $fc1, name=>'relu1', act_type=>"relu");
    }
    my %set_stage1 = map { $_ => 1 } @{ $act1->list_arguments };

    my ($fc2, $act2, $fc3, $mlp);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage2');
        $fc2  = mx->symbol->FullyConnected(data => $act1, name => 'fc2', num_hidden => 64);
        $act2 = mx->symbol->Activation(data => $fc2, name=>'relu2', act_type=>"relu");
        $fc3  = mx->symbol->FullyConnected(data => $act2, name=>'fc3', num_hidden=>10);
        $fc3  = mx->symbol->BatchNorm($fc3);
        $mlp  = mx->symbol->SoftmaxOutput(data => $fc3, name => 'softmax');
    }
    my %set_stage2 = map { $_ => 1 } @{ $mlp->list_arguments };
    for my $k (keys %set_stage1)
    {
        delete $set_stage2{$k};
    }

    my $group2ctx = {
        stage1 => mx->cpu(1),
        stage2 => mx->cpu(2)
    };

    my $texec = $mlp->simple_bind(
        ctx       => mx->cpu(0),
        group2ctx => $group2ctx,
        shapes    => { data => [1,200] }
    );

    for(zip($texec->arg_arrays, $mlp->list_arguments())) {
        my ($arr, $name) = @$_;
        if(exists $set_stage1{ $name })
        {
            ok($arr->context == $group2ctx->{stage1});
        }
        else
        {
            ok($arr->context == $group2ctx->{stage2});
        }
    }
}

test_ctx_group();
