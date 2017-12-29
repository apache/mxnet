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
use Test::More tests => 119;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::TestUtils qw(almost_equal);
use Scalar::Util qw(refaddr);
use AI::MXNet::Base;

sub test_parameter
{
    my $p = gluon->Parameter('weight', shape=>[10, 10]);
    $p->initialize(init=>'xavier', ctx=>[mx->cpu(0), mx->cpu(1)]);
    ok(@{$p->list_data} == 2);
    ok(@{$p->list_grad} == 2);
    ok($p->data(mx->cpu(1))->context eq mx->cpu(1));
    is_deeply($p->data(mx->cpu(0))->shape, [10, 10]);
    ok($p->var->name eq  'weight');

    $p->reset_ctx(ctx=>[mx->cpu(1), mx->cpu(2)]);
    is_deeply($p->list_ctx, [mx->cpu(1), mx->cpu(2)]);
}

test_parameter();

sub test_paramdict
{
    my $params = gluon->ParameterDict('net_');
    $params->get('weight', shape=>[10, 10]);
    is_deeply([$params->keys], ['net_weight']);
    $params->initialize(ctx=>mx->cpu());
    $params->save('test.params');
    $params->load('test.params', ctx => mx->cpu());
}

test_paramdict();

package Net;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Function::Parameters;
extends 'AI::MXNet::Gluon::Block';

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->dense0(nn->Dense(5, in_units=>5));
        $self->dense1(nn->Dense(5, in_units=>5));
    });
}

method forward($x)
{
    return $self->dense1->($self->dense0->($x));
}

package main;

sub test_parameter_sharing
{
    my $net1 = Net->new(prefix=>'net1_');
    my $net2 = Net->new(prefix=>'net2_', params=>$net1->collect_params());
    $net1->collect_params()->initialize();
    $net2->(mx->nd->zeros([3, 5]));
    $net1->save_params('net1.params');
    my $net3 = Net->new(prefix=>'net3_');
    $net3->load_params('net1.params', ctx => mx->cpu());
}

test_parameter_sharing();

sub test_basic
{
    my $model = nn->Sequential();
    $model->add(nn->Dense(128, activation=>'tanh', in_units=>10, flatten=>0));
    $model->add(nn->Dropout(0.5));
    $model->add(nn->Dense(64, activation=>'tanh', in_units=>256));
    $model->add(nn->Dense(32, in_units=>64));
    $model->add(nn->Activation('relu'));

    # symbol
    my $x = mx->sym->var('data');
    my $y = $model->($x);
    ok(@{ $y->list_arguments } == 7);

    # ndarray
    $model->collect_params()->initialize(init => mx->init->Xavier(magnitude=>2.24));
    $x = $model->(mx->nd->zeros([32, 2, 10]));
    is_deeply($x->shape, [32, 32]);
    $x->wait_to_read;

    $model->collect_params()->setattr(grad_req => 'null');
    ok(not defined( ($model->collect_params()->values())[0]->_grad));
    $model->collect_params()->setattr(grad_req => 'write');
    ok(defined (($model->collect_params()->values())[0]->_grad));
}

test_basic();

sub test_dense
{
    my $model = nn->Dense(128, activation=>'tanh', in_units=>10, flatten=>0, prefix=>'test_');
    my $inputs = mx->sym->Variable('data');
    my $outputs = $model->($inputs);
    is_deeply({map { $_ => 1 } $model->collect_params()->keys()}, {'test_weight', 1, 'test_bias', 1});
    is_deeply($outputs->list_outputs(), ['test_tanh_fwd_output']);
    my ($args, $outs, $auxs) = $outputs->infer_shape(data=>[2, 3, 10]);
    is_deeply($outs, [[2, 3, 128]]);

    $model = nn->Dense(128, activation=>'relu', in_units=>30, flatten=>1, prefix=>'test2_');
    $inputs = mx->sym->Variable('data');
    $outputs = $model->($inputs);
    is_deeply({map { $_ => 1 } $model->collect_params()->keys()}, {'test2_weight', 1, 'test2_bias', 1});
    is_deeply($outputs->list_outputs(), ['test2_relu_fwd_output']);
    ($args, $outs, $auxs) = $outputs->infer_shape(data=>[17, 2, 5, 3]);
    is_deeply($outs, [[17, 128]]);
}

test_dense();

package Net2;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Function::Parameters;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'model' => (is => 'rw');

method hybrid_forward($F, $x)
{
    my $out = $self->model->($x);
    return $F->add_n(map { $_->sum } @{ $out });
}

package main;

sub test_symbol_block
{
    my $model = nn->HybridSequential();
    $model->add(nn->Dense(128, activation=>'tanh'));
    $model->add(nn->Dropout(0.5));
    $model->add(nn->Dense(64, activation=>'tanh'));
    $model->add(nn->Dense(32, in_units=>64));
    $model->add(nn->Activation('relu'));

    $model->initialize();

    my $inputs = mx->sym->var('data');
    my $outputs = $model->($inputs)->get_internals();
    my $smodel = gluon->SymbolBlock($outputs, $inputs, params=>$model->collect_params);

    ok(@{ $smodel->(mx->nd->zeros([16, 10])) } == 14);
    my $out = $smodel->(mx->sym->var('in'));
    ok(@{ $out } == @{ $outputs->list_outputs() });

    my $net = Net2->new(model => $smodel);
    $net->hybridize();
    ok(ref $net->(mx->nd->zeros([16, 10])) eq 'AI::MXNet::NDArray');

    $inputs = mx->sym->var('data');
    $outputs = $model->($inputs);
    $smodel = gluon->SymbolBlock($outputs, $inputs, params=>$model->collect_params);
    $net = Net2->new(model => $smodel);
    $net->hybridize();
    ok(ref $net->(mx->nd->zeros([16, 10])) eq 'AI::MXNet::NDArray');
}

test_symbol_block();

sub check_layer_forward
{
    my ($layer, $dshape) = @_;
    $layer->collect_params()->initialize();
    my $x = mx->nd->ones($dshape);
    $x->attach_grad();
    my $out;
    mx->autograd->record(sub {
        $out = $layer->($x);
    });
    $out->backward();
    my $pdl_out = $out->aspdl;
    my $pdl_dx  = $x->grad->aspdl;

    $layer->hybridize();

    $x = mx->nd->ones($dshape);
    $x->attach_grad();
    mx->autograd->record(sub {
        $out = $layer->($x);
    });
    $out->backward();

    ok(almost_equal($pdl_out, $out->aspdl, 1e-5));
    ok(almost_equal($pdl_dx, $x->grad->aspdl, 1e-5));
}

sub test_conv
{
    my @layers1d = (
        nn->Conv1D(16, 3, in_channels=>4),
        nn->Conv1D(16, 3, groups=>2, in_channels=>4),
        nn->Conv1D(16, 3, strides=>3, groups=>2, in_channels=>4),
    );
    for my $layer (@layers1d)
    {
        check_layer_forward($layer, [1, 4, 10]);
    }

    my @layers2d = (
        nn->Conv2D(16, [3, 4], in_channels=>4),
        nn->Conv2D(16, [5, 4], in_channels=>4),
        nn->Conv2D(16, [3, 4], groups=>2, in_channels=>4),
        nn->Conv2D(16, [3, 4], strides=>4, in_channels=>4),
        nn->Conv2D(16, [3, 4], dilation=>4, in_channels=>4),
        nn->Conv2D(16, [3, 4], padding=>4, in_channels=>4),
    );
    for my $layer (@layers2d)
    {
        check_layer_forward($layer, [1, 4, 20, 20]);
    }

    my @layers3d = (
        nn->Conv3D(16, [1, 8, 4], in_channels=>4, activation=>'relu'),
        nn->Conv3D(16, [5, 4, 3], in_channels=>4),
        nn->Conv3D(16, [3, 3, 3], groups=>2, in_channels=>4),
        nn->Conv3D(16, 4, strides=>4, in_channels=>4),
        nn->Conv3D(16, [3, 3, 3], padding=>4, in_channels=>4),
    );
    for my $layer (@layers3d)
    {
        check_layer_forward($layer, [1, 4, 10, 10, 10]);
    }

    # These layouts only supported on GPU for now
    my $layer = nn->Conv2D(16, [3, 3], layout=>'NHWC', in_channels=>4);
    #check_layer_forward($layer, [1, 10, 10, 4]);

    $layer = nn->Conv3D(16, [3, 3, 3], layout=>'NDHWC', in_channels=>4);
    # check_layer_forward(layer, (1, 10, 10, 10, 4))
}

test_conv();


sub test_deconv
{
    # commented out code is only supported on GPU for now
    # my @layers1d = (
    #     nn->Conv1DTranspose(16, 3, in_channels=>4),
    #     nn->Conv1DTranspose(16, 3, groups=>2, in_channels=>4),
    #     nn->Conv1DTranspose(16, 3, strides=>3, groups=>2, in_channels=>4),
    # );
    # for my $layer (@layers1d)
    # {
    #     check_layer_forward($layer, [1, 4, 10]);
    # }


    my @layers2d = (
        nn->Conv2DTranspose(16, [3, 4], in_channels=>4),
        nn->Conv2DTranspose(16, [5, 4], in_channels=>4),
        nn->Conv2DTranspose(16, [3, 4], groups=>2, in_channels=>4),
        nn->Conv2DTranspose(16, [3, 4], strides=>4, in_channels=>4),
        nn->Conv2DTranspose(16, [3, 4], dilation=>4, in_channels=>4),
        nn->Conv2DTranspose(16, [3, 4], padding=>4, in_channels=>4),
        nn->Conv2DTranspose(16, [3, 4], strides=>4, output_padding=>3, in_channels=>4),
    );
    for my $layer (@layers2d)
    {
        check_layer_forward($layer, [1, 4, 20, 20]);
    }

    # @layers3d = (
    #     nn->Conv3DTranspose(16, [1, 8, 4], in_channels=>4),
    #     nn->Conv3DTranspose(16, [5, 4, 3], in_channels=>4),
    #     nn->Conv3DTranspose(16, [3, 3, 3], groups=>2, in_channels=>4),
    #     nn->Conv3DTranspose(16, 4, strides=>4, in_channels=>4),
    #     nn->Conv3DTranspose(16, [3, 3, 3], padding=>4, in_channels=>4),
    # );
    # for my $layer (@layers3d)
    # {
    #     check_layer_forward($layer, [1, 4, 10, 10, 10]);
    # }
    #
    my $layer = nn->Conv2DTranspose(16, [3, 3], layout=>'NHWC', in_channels=>4);
    # check_layer_forward($layer, [1, 10, 10, 4]);
    #
    # $layer = nn->Conv3DTranspose(16, [3, 3, 3], layout=>'NDHWC', in_channels=>4);
    # check_layer_forward(layer, [1, 10, 10, 10, 4]);
}

test_deconv();

sub test_pool
{
    my @layers1d = (
        nn->MaxPool1D(),
        nn->MaxPool1D(3),
        nn->MaxPool1D(3, 2),
        nn->AvgPool1D(),
        nn->GlobalAvgPool1D(),
    );
    for my $layer (@layers1d)
    {
        check_layer_forward($layer, [1, 2, 10]);
    }

    my @layers2d = (
        nn->MaxPool2D(),
        nn->MaxPool2D([3, 3]),
        nn->MaxPool2D(3, 2),
        nn->AvgPool2D(),
        nn->GlobalAvgPool2D(),
    );
    for my $layer (@layers2d)
    {
        check_layer_forward($layer, [1, 2, 10, 10]);
    }

    my @layers3d = (
        nn->MaxPool3D(),
        nn->MaxPool3D([3, 3, 3]),
        nn->MaxPool3D(3, 2),
        nn->AvgPool3D(),
        nn->GlobalAvgPool3D(),
    );
    for my $layer (@layers3d)
    {
        check_layer_forward($layer, [1, 2, 10, 10, 10]);
    }

    # test ceil_mode
    my $x = mx->nd->zeros([2, 2, 10, 10]);

    my $layer = nn->MaxPool2D(3, ceil_mode=>0);
    $layer->collect_params()->initialize();
    is_deeply($layer->($x)->shape, [2, 2, 3, 3]);

    $layer = nn->MaxPool2D(3, ceil_mode=>1);
    $layer->collect_params()->initialize();
    is_deeply($layer->($x)->shape, [2, 2, 4, 4]);
}

test_pool();

sub test_batchnorm
{
    my $layer = nn->BatchNorm(in_channels=>10);
    check_layer_forward($layer, [2, 10, 10, 10]);
}

test_batchnorm();

sub test_reshape
{
    my $x = mx->nd->ones([2, 4, 10, 10]);
    my $layer = nn->Conv2D(10, 2, in_channels=>4);
    $layer->collect_params()->initialize();
    mx->autograd->record(sub {
        $x = $layer->($x);
        $x = $x->reshape([-1]);
        $x = $x + 10;
    });
    $x->backward();
}

test_reshape();

sub test_slice
{
    my $x = mx->nd->ones([5, 4, 10, 10]);
    my $layer = nn->Conv2D(10, 2, in_channels=>4);
    $layer->collect_params()->initialize();
    mx->autograd->record(sub {
        $x = $layer->($x);
        $x = $x->slice([1,3]);
        $x = $x + 10;
    });
    $x->backward();
}

test_slice();

sub test_at
{
    my $x = mx->nd->ones([5, 4, 10, 10]);
    my $layer = nn->Conv2D(10, 2, in_channels=>4);
    $layer->collect_params()->initialize();
    mx->autograd->record(sub {
        $x = $layer->($x);
        $x = $x->at(1);
        $x = $x + 10;
    });
    $x->backward();
}

test_at();

sub test_deferred_init
{
    my $x = mx->nd->ones([5, 4, 10, 10]);
    my $layer = nn->Conv2D(10, 2);
    $layer->collect_params()->initialize();
    $layer->($x);
}

test_deferred_init();


sub check_split_data
{
    my ($x, $num_slice, $batch_axis, %kwargs) = @_;
    my $res = gluon->utils->split_data($x, $num_slice, $batch_axis, %kwargs);
    ok(@{ $res } == $num_slice);
    ok(almost_equal(mx->nd->concat(@$res, dim=>$batch_axis)->aspdl(), $x->aspdl()));
}

sub test_split_data
{
    my $x = mx->nd->random->uniform(shape=>[128, 33, 64]);

    check_split_data($x, 8, 0);
    check_split_data($x, 3, 1);
    check_split_data($x, 4, 1, even_split=>0);
    check_split_data($x, 15, 1, even_split=>0);
    eval {
        check_split_data($x, 4, 1);
    };
    ok($@);
}

test_split_data();

sub test_flatten
{
    my $flatten = nn->Flatten();
    my $x = mx->nd->zeros([3,4,5,6]);
    is_deeply($flatten->($x)->shape, [3, 4*5*6]);
    $x = mx->nd->zeros([3,6]);
    is_deeply($flatten->($x)->shape, [3, 6]);
    $x = mx->nd->zeros([3]);
    is_deeply($flatten->($x)->shape, [3, 1]);
}

test_flatten();

sub test_trainer
{
    my $dict_equ = sub { my ($a, $b) = @_;
        is_deeply({ map { $_ => 1 } keys %$a }, { map { $_ => 1 } keys %$b });
        for my $k (keys %$a)
        {
            ok(($a->{$k}->aspdl == $b->{$k}->aspdl)->all);
        }
    };
    my $x = gluon->Parameter('x', shape=>[10]);
    $x->initialize(ctx=>[mx->cpu(0), mx->cpu(1)], init=>'zeros');
    my $trainer = gluon->Trainer([$x], 'sgd', {'learning_rate'=> 1.0, 'momentum'=> 0.5});
    my $y;
    mx->autograd->record(sub {
        for my $w (@{ $x->list_data() })
        {
            $y = $w + 1;
            $y->backward();
        }
    });
    $trainer->step(1);

    ok(($x->data(mx->cpu(1))->aspdl == -2)->all);

    $x->lr_mult(0.5);

    mx->autograd->record(sub {
        for my $w (@{ $x->list_data() })
        {
            $y = $w + 1;
            $y->backward();
        }
    });
    $trainer->step(1);

    ok(($x->data(mx->cpu(1))->aspdl == -4)->all);

    $trainer->save_states('test.states');
    my $states;
    if($trainer->_update_on_kvstore)
    {
        $states = { %{ $trainer->_kv_store->_updater->states } };
    }
    else
    {
        $states = { %{ $trainer->_updaters->[0]->states } };
    }
    $trainer->load_states('test.states');
    if($trainer->_update_on_kvstore)
    {
        $dict_equ->($trainer->_kv_store->_updater->states, $states);
        ok($trainer->_optimizer eq $trainer->_kv_store->_updater->optimizer);
    }
    else
    {
        for my $updater (@{ $trainer->_updaters })
        {
            $dict_equ->($updater->states, $states);
        }
        ok($trainer->_optimizer eq $trainer->_updaters->[0]->optimizer);
    }
}

test_trainer();

sub test_block_attr_hidden
{
    my $b = gluon->Block();
    # regular attributes can change types
    $b->a(undef);
    $b->a(1);
}

test_block_attr_hidden();

sub test_block_attr_block
{
    my $b = gluon->Block();
    # regular variables can't change types
    $b->b(gluon->Block());
    eval { $b->b([2]); };
    ok($@ =~ /not allowed/i);
}

test_block_attr_block();

sub test_block_attr_param
{
    my $b = gluon->Block();
    # regular variables can't change types
    $b->b(gluon->Parameter(name => 'test'));
    eval { $b->b([2]); };
    ok($@ =~ /not allowed/i);
}

test_block_attr_param();

sub test_block_attr_regular
{
    my $b = gluon->Block();

    # set block attribute also sets _children
    $b->c(gluon->Block());
    my $c2 = gluon->Block();
    $b->c($c2);
    ok(refaddr($b->c) == refaddr($c2) and refaddr($b->_children->[0]) == refaddr($c2));
}

test_block_attr_regular();

sub test_embedding
{
    my $layer = gluon->nn->Embedding(10, 100);
    $layer->initialize();
    my $x = mx->nd->array([3,4,2,0,1]);
    my $y;
    mx->autograd->record(sub {
        $y = $layer->($x);
        $y->backward();
    });
    ok(($layer->weight->grad->slice([0,4]) == 1)->aspdl->all);
    ok(($layer->weight->grad->slice([5, -1]) == 0)->aspdl->all);
}

test_embedding();
