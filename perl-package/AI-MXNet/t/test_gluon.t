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
use Test::More tests => 232;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::TestUtils qw(almost_equal dies_ok);
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
    ok($p->grad(mx->cpu(0))->stype eq 'default');
    ok($p->data(mx->cpu(0))->stype eq 'default');

    $p->reset_ctx(ctx=>[mx->cpu(1), mx->cpu(2)]);
    is_deeply($p->list_ctx, [mx->cpu(1), mx->cpu(2)]);
}

test_parameter();

sub test_invalid_parameter_stype
{
    dies_ok(sub { gluon->Parameter('weight', shape=>[10, 10], stype=>'invalid') });
}

test_invalid_parameter_stype();

sub test_invalid_parameter_grad_stype
{
    dies_ok(sub { gluon->Parameter('weight', shape=>[10, 10], grad_stype=>'invalid') });
}

test_invalid_parameter_grad_stype();

sub test_sparse_parameter
{
    my $p = gluon->Parameter('weight', shape=>[10, 10], stype=>'row_sparse', grad_stype=>'row_sparse');
    $p->initialize(init=>'xavier', ctx=>[mx->cpu(0), mx->cpu(1)]);
    my $row_id = mx->nd->arange(start => 0, stop => 10, ctx=>mx->cpu(1));
    ok(@{ $p->list_grad } == 2);
    # getting row_sparse data without trainer throws an exception
    dies_ok(sub { $p->list_row_sparse_data($row_id) });
    my $trainer = gluon->Trainer([$p], 'sgd');
    ok(@{ $p->list_row_sparse_data($row_id) } == 2);
    my $weight = $p->row_sparse_data($row_id);
    ok($weight->context eq mx->cpu(1));
    is_deeply($weight->shape, [10, 10]);
    ok($weight->stype eq 'row_sparse');
    ok($p->var->name eq 'weight');
    ok($p->var->attr('__storage_type__') eq STORAGE_TYPE_STR_TO_ID->{row_sparse});
    ok($p->grad(mx->cpu(0))->stype eq 'row_sparse');

    $p->reset_ctx(ctx=>[mx->cpu(1), mx->cpu(2)]);
    is_deeply($p->list_ctx, [mx->cpu(1), mx->cpu(2)]);
}

test_sparse_parameter();

sub test_parameter_invalid_access
{
    # cannot call data on row_sparse parameters
    my $p0 = gluon->Parameter('weight', shape=>[10, 10], stype=>'row_sparse', grad_stype=>'row_sparse');
    $p0->initialize(init=>'xavier', ctx=>[mx->cpu(0), mx->cpu(1)]);
    dies_ok(sub { $p0->data });
    dies_ok(sub { $p0->list_data });
    my $row_id = mx->nd->arange(start => 0, stop => 10);
    # cannot call row_sparse_data on dense parameters
    my $p1 = gluon->Parameter('weight', shape=>[10, 10]);
    $p1->initialize(init=>'xavier', ctx=>[mx->cpu(0), mx->cpu(1)]);
    dies_ok(sub { $p1->row_sparse_data($row_id->copyto(mx->cpu(0))) });
    dies_ok(sub { $p1->list_row_sparse_data($row_id) });
}

test_parameter_invalid_access();

sub test_paramdict
{
    my $ctx = mx->cpu(1);
    my $params0 = gluon->ParameterDict('net_');
    $params0->get('w0', shape=>[10, 10]);
    $params0->get('w1', shape=>[10, 10], stype=>'row_sparse');
    my $all_row_ids = mx->nd->arange(start => 0, stop => 10, ctx=>$ctx);
    # check param names
    is_deeply([$params0->keys()], ['net_w0', 'net_w1']);
    $params0->initialize(ctx=>$ctx);
    my $trainer0 = gluon->Trainer($params0, 'sgd');
    my $prev_w0 = $params0->get('w0')->data($ctx);
    my $prev_w1 = $params0->get('w1')->row_sparse_data($all_row_ids);
    # save params
    $params0->save('test_paramdict.params');

    # load params
    my $params1 = gluon->ParameterDict('net_');
    $params1->get('w0', shape=>[10, 10]);
    $params1->get('w1', shape=>[10, 10], stype=>'row_sparse');
    $params1->load('test_paramdict.params', ctx=>$ctx);
    my $trainer1 = gluon->Trainer($params1, 'sgd');

    # compare the values before and after save/load
    my $cur_w0 = $params1->get('w0')->data($ctx);
    my $cur_w1 = $params1->get('w1')->row_sparse_data($all_row_ids);
    ok(almost_equal($prev_w0->aspdl, $cur_w0->aspdl));
    ok(almost_equal($prev_w1->aspdl, $cur_w1->aspdl));

    # create a new param dict with dense params, and load from the checkpoint
    # of sparse & dense params
    my $params2 = gluon->ParameterDict('net_');
    $params2->get('w0', shape=>[10, 10]);
    $params2->get('w1', shape=>[10, 10]);
    $params2->load('test_paramdict.params', ctx=>$ctx);

    # compare the values before and after save/load
    $cur_w0 = $params2->get('w0')->data($ctx);
    $cur_w1 = $params2->get('w1')->data($ctx);
    ok(almost_equal($prev_w0->aspdl, $cur_w0->aspdl));
    ok(almost_equal($prev_w1->aspdl, $cur_w1->aspdl));
}

test_paramdict();

sub test_parameter_row_sparse_data
{
    my $ctx0 = mx->cpu(1);
    my $ctx1 = mx->cpu(2);
    my $dim0 = 4;
    my $x = gluon->Parameter('x', shape=>[$dim0, 2], stype=>'row_sparse');
    $x->initialize(init=>'xavier', ctx=>[$ctx0, $ctx1]);
    my $trainer = gluon->Trainer([$x], 'sgd');
    my $x_param = $x->_data->[0]->copy();
    is($x_param->stype, 'row_sparse');
    my $row_id_0 = mx->nd->array([0,1], ctx=>$ctx0);
    my $retained_0 = $x->row_sparse_data($row_id_0);
    my $retained_target_0 = mx->nd->sparse->retain($x_param, $row_id_0->as_in_context($ctx0));
    ok(almost_equal($retained_0->aspdl, $retained_target_0->aspdl));
    is($retained_0->context, $ctx0);
    my $row_id_1 = mx->nd->arange(start => 0, stop => $dim0, ctx=>$ctx1);
    my $retained_1 = $x->row_sparse_data($row_id_1);
    my $retained_target_1 = $x_param;
    ok(almost_equal($retained_1->aspdl, $retained_target_1->aspdl));
    is($retained_1->context, $ctx1);
    my $row_id_2 = mx->nd->array([0,1,2]);
    my $retained_2 = $x->list_row_sparse_data($row_id_2);
    my $retained_target_2 = mx->nd->sparse->retain($x_param, $row_id_2->as_in_context($ctx0));
    ok(almost_equal($retained_2->[0]->aspdl, $retained_target_2->aspdl));
}

test_parameter_row_sparse_data();

sub test_constant
{
    package Test {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::HybridBlock';
        sub BUILD
        {
            my $self = shift;
            $self->value(mx->nd->array([[1,2], [3,4]])->aspdl);
            $self->const($self->params->get_constant('const', $self->value));
        }
        sub hybrid_forward
        {
            my ($self, $F, $x, $name, $const) = @_;
            return $x + $const;
        }
    };

    my $test = Test->new();
    $test->initialize();
    my $trainer = gluon->Trainer(
        $test->collect_params(), 'sgd',
        {learning_rate => 1.0, momentum => 0.5}
    );

    my ($x, $y);
    mx->autograd->record(sub {
        $x = mx->nd->ones([2,2]);
        $x->attach_grad();
        $y = $test->($x);
        $y->backward();
    });

    $trainer->step(1);

    ok(($test->const->data->aspdl == $test->value)->all);
    ok(($x->grad->aspdl == 1)->all);
}

test_constant();

package Net;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Function::Parameters;
extends 'AI::MXNet::Gluon::Block';
has 'in_units' => (is => 'rw', default => 0);

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->dense0(nn->Dense(5, in_units=>$self->in_units));
        $self->dense1(nn->Dense(5, in_units=>$self->in_units));
    });
}

method forward($x)
{
    return $self->dense1->($self->dense0->($x));
}

package main;

sub test_parameter_sharing
{
    my $net1 = Net->new(prefix=>'net1_', in_units => 5);
    my $net2 = Net->new(prefix=>'net2_', params=>$net1->collect_params());
    $net1->collect_params()->initialize();
    $net2->(mx->nd->zeros([3, 5]));
    $net1->save_parameters('net1.params');
    my $net3 = Net->new(prefix=>'net3_');
    $net3->load_parameters('net1.params', ctx => mx->cpu());
    my $net4 = Net->new(prefix=>'net4_');
    my $net5 = Net->new(prefix=>'net5_', in_units=>5, params=>$net4->collect_params());
    $net4->collect_params()->initialize();
    $net5->(mx->nd->zeros([3, 5]));
    $net4->save_parameters('net4.params');
    my $net6 = Net->new(prefix=>'net6_');
    $net6->load_parameters('net4.params', ctx => mx->cpu());
}

test_parameter_sharing();

sub test_parameter_str
{
    package Net1 {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            my $self = shift;
            $self->name_scope(sub {
                $self->dense0(nn->Dense(10, in_units=>5, use_bias=>0));
            });
        }
    };
    my $net = Net1->new(prefix=>'net1_');
    my @lines = split(/\n/, $net->collect_params());
    ok($lines[0] eq 'net1_ (');
    ok($lines[1] =~ /net1_dense0_weight/);
    ok($lines[1] =~ /\(10, 5\)/);
    ok($lines[1] =~ /float32/);
    ok($lines[2] eq ')');
}

test_parameter_str();

sub test_collect_parameters
{
    my $net = nn->HybridSequential(prefix=>"test_");
    $net->name_scope(sub {
        $net->add(nn->Conv2D(10, 3));
        $net->add(nn->Dense(10, activation=>'relu'));
    });
    is_deeply(
        [$net->collect_params->keys],
        ['test_conv0_weight', 'test_conv0_bias','test_dense0_weight','test_dense0_bias']
    );
    is_deeply(
        [$net->collect_params('.*weight')->keys],
        ['test_conv0_weight', 'test_dense0_weight']
    );
    is_deeply(
        [$net->collect_params('test_conv0_bias|test_dense0_bias')->keys],
        ['test_conv0_bias', 'test_dense0_bias']
    )
};

test_collect_parameters();

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

    ok($smodel->(mx->nd->zeros([16, 10])) == 14);
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

sub test_sparse_symbol_block
{
    my $data = mx->sym->var('data');
    my $weight = mx->sym->var('weight', stype=>'row_sparse');
    my $bias = mx->sym->var('bias');
    my $out = mx->sym->broadcast_add(mx->sym->dot($data, $weight), $bias);
    # an exception is expected when creating a SparseBlock w/ sparse param
    dies_ok(sub { gluon->SymbolBlock($out, $data) });
}

test_sparse_symbol_block();

sub test_sparse_hybrid_block0
{
    my $params = gluon->ParameterDict('net_');
    $params->get('weight', shape=>[5,5], stype=>'row_sparse', dtype=>'float32', allow_deferred_init => 1);
    $params->get('bias', shape=>[5], dtype=>'float32', allow_deferred_init => 1);
    my $net = nn->Dense(5, params=>$params);
    $net->initialize();
    my $x = mx->nd->ones([2,5]);
    # an exception is expected when forwarding a HybridBlock w/ sparse param
    dies_ok(sub { $net->($x) });
}

test_sparse_hybrid_block0();

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
        nn->AvgPool1D(count_include_pad=>0),
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
        nn->AvgPool2D(count_include_pad=>0),
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
        nn->AvgPool3D(count_include_pad=>0),
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

sub test_instancenorm
{
    my $layer = nn->InstanceNorm(in_channels=>10);
    check_layer_forward($layer, [2, 10, 10, 10]);
}

test_instancenorm();

sub test_layernorm
{
    my $layer = nn->LayerNorm(in_channels=>10);
    check_layer_forward($layer, [2, 10, 10, 10]);
}

test_layernorm();

sub test_reflectionpad
{
    my $layer = nn->ReflectionPad2D(3);
    check_layer_forward($layer, [2, 3, 24, 24]);
}

test_reflectionpad();

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
    ok(refaddr($b->c) == refaddr($c2) and refaddr(($b->_children->values)[0]) == refaddr($c2));
}

test_block_attr_regular();

sub test_block_attr_list_of_block
{
    package Model1 {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            my $self = shift;
            $self->name_scope(sub {
                $self->layers([map { nn->Dense($_ * 10) } 0..5]);
            });
        }
    };
    package Model2 {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            my $self = shift;
            $self->name_scope(sub {
                $self->layers({});
                $self->layers->{a} = [map { nn->Dense($_ * 10) } 0..5];
            });
        }
    };
    package Model3 {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            my $self = shift;
            $self->name_scope(sub {
                $self->layers(nn->Sequential());
                $self->layers->add(map { nn->Dense($_ * 10) } 0..5);
            });
        }
    };
    package Model4 {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            my $self = shift;
            $self->name_scope(sub {
                $self->data({a => '4', b => 123});
            });
        }
    };
    my $w = 0;
    local($SIG{__WARN__}) = sub {
        $w++;
    };
    Model1->new->collect_params;
    ok($w > 0); $w = 0;
    Model2->new->collect_params;
    ok($w > 0); $w = 0;
    Model3->new->collect_params;
    ok($w == 0); $w = 0;
    Model4->new->collect_params;
    ok($w == 0);
}

test_block_attr_list_of_block();

sub check_sequential
{
    my ($net) = @_;
    my $dense1 = nn->Dense(10);
    $net->add($dense1);
    my $dense2 = nn->Dense(10);
    $net->add($dense2);
    my $dense3 = nn->Dense(10);
    $net->add($dense3);

    ok(refaddr($net->[1]) == refaddr($dense2));
    ok(refaddr($net->[-1]) == refaddr($dense3));
    my $slc = $net->slice([1,2]);
    ok(@$slc == 2 and refaddr($slc->[0]) == refaddr($dense2) and refaddr($slc->[1]) == refaddr($dense3));
    ok(ref $slc eq ref $net);
}

sub test_sequential
{
    check_sequential(nn->Sequential());
    check_sequential(nn->HybridSequential());
}

test_sequential();

sub test_global_norm_clip
{
    my @stypes = ('default', 'row_sparse');
    my $check_global_norm_clip = sub { my ($stype) = @_;
        my $x1 = mx->nd->ones([3,3])->tostype($stype);
        my $x2 = mx->nd->ones([4,4])->tostype($stype);
        my $norm = gluon->utils->clip_global_norm([$x1, $x2], 1.0);
        ok($norm == 5);
        ok(almost_equal($x1->aspdl, mx->nd->ones([3,3])->aspdl/5));
        ok(almost_equal($x2->aspdl, mx->nd->ones([4,4])->aspdl/5));

        my $x3 = mx->nd->array([1.0, 2.0, 'nan'])->tostype($stype);
        my $w = 0;
        local($SIG{__WARN__}) = sub {
            $w++;
        };
        gluon->utils->clip_global_norm([$x1, $x3], 2.0);
        ok($w == 1);
    };
    for my $stype (@stypes)
    {
        $check_global_norm_clip->($stype);
    }
}

test_global_norm_clip();

sub test_embedding
{
    local($ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE}) = 0;
    my $check_embedding = sub { my ($sparse_grad) = @_;
        my $layer = nn->Embedding(10, 100, sparse_grad=>$sparse_grad);
        $layer->initialize();
        my $x = mx->nd->array([3,4,2,0,1]); my $y;
        mx->autograd->record(sub {
            $y = $layer->($x);
            $y->backward();
        });
        ok(($layer->weight->grad->aspdl->slice('X', [0, 4]) == 1)->all);
        ok(($layer->weight->grad->aspdl->slice('X', [5, -1]) == 0)->all);
    };
    my $check_embedding_large_input = sub { my ($sparse_grad) = @_;
        my $embedding = nn->Embedding(10, 1, sparse_grad=>$sparse_grad);
        $embedding->initialize();
        $embedding->hybridize();
        my $shape = [20481];
        my ($emb_in, $loss);
        mx->autograd->record(sub {
            $emb_in = $embedding->(mx->nd->ones($shape));
            $loss = $emb_in->sum;
        });
        $loss->backward;
        ok($embedding->weight->grad->sum->asscalar == 20481);
    };
    $check_embedding->(1);
    $check_embedding->(0);
    $check_embedding_large_input->(1);
    $check_embedding_large_input->(0);
}

test_embedding();

sub test_hybrid_stale_cache
{
    my $net = nn->HybridSequential();
    $net->name_scope(sub {
        $net->add(nn->Dense(10, weight_initializer=>'zeros', bias_initializer=>'ones', flatten=>0));
    });

    $net->hybridize();
    $net->initialize();
    $net->(mx->nd->ones([2,3,5]));

    $net->add(nn->Flatten());
    is_deeply($net->(mx->nd->ones([2,3,5]))->shape, [2, 30]);

    $net = nn->HybridSequential();
    $net->name_scope(sub {
        $net->fc1(nn->Dense(10, weight_initializer=>'zeros',
                                    bias_initializer=>'ones', flatten=>0));
        $net->fc2(nn->Dense(10, weight_initializer=>'zeros',
                                    bias_initializer=>'ones', flatten=>0));
    });
    $net->hybridize();
    $net->initialize();
    $net->(mx->nd->ones([2,3,5]));

    $net->fc2(nn->Dense(10, weight_initializer=>'zeros',
                                bias_initializer=>'ones', flatten=>1));
    $net->initialize();
    is_deeply($net->(mx->nd->ones([2,3,5]))->shape, [2, 10]);
}

test_hybrid_stale_cache();

sub test_lambda
{
    my $net1 = nn->HybridSequential();
    $net1->add(nn->Activation('tanh'),
             nn->LeakyReLU(0.1));

    my $net2 = nn->HybridSequential();
    my $op3 = sub { my ($F, $x, @args) = @_; $F->LeakyReLU($x, @args, slope=>0.1); };
    $net2->add(nn->HybridLambda('tanh'),
             nn->HybridLambda($op3));

    my $op4 = sub { mx->nd->LeakyReLU($_[0], slope=>0.1); };
    my $net3 = nn->Sequential();
    $net3->add(nn->Lambda('tanh'),
             nn->Lambda($op4));

    my $input_data = mx->nd->random->uniform(shape=>[2, 3, 5, 7]);
    my ($out1, $out2, $out3) = ($net1->($input_data), $net2->($input_data), $net3->($input_data));
    ok(almost_equal($out1->aspdl, $out2->aspdl, 1e-3));
    ok(almost_equal($out1->aspdl, $out3->aspdl, 1e-3));
}

test_lambda();

sub test_fill_shape_deferred
{
    my $net = nn->HybridSequential();
    $net->name_scope(sub {
        $net->add(nn->Conv2D(64, kernel_size=>2, padding=>1),
                nn->BatchNorm(),
                nn->Dense(10));
    });
    $net->hybridize();
    $net->initialize();
    $net->(mx->nd->ones([2,3,5,7]));
    ok($net->[0]->weight->shape->[1] == 3);
    ok($net->[1]->gamma->shape->[0] == 64);
    ok($net->[2]->weight->shape->[1] == 3072);
}

test_fill_shape_deferred();

sub test_fill_shape_load
{
    my $ctx = mx->context->current_context();
    my $net1 = nn->HybridSequential();
    $net1->name_scope(sub {
        $net1->add(nn->Conv2D(64, kernel_size=>2, padding=>1),
                 nn->BatchNorm(),
                 nn->Dense(10))
    });
    $net1->hybridize();
    $net1->initialize(mx->init->Uniform, ctx => $ctx);
    $net1->(mx->nd->ones([2,3,5,7], ctx => $ctx));
    $net1->save_parameters('net_fill.params');

    my $net2 = nn->HybridSequential();
    $net2->name_scope(sub {
        $net2->add(nn->Conv2D(64, kernel_size=>2, padding=>1),
                 nn->BatchNorm(),
                 nn->Dense(10))
    });
    $net2->hybridize();
    $net2->initialize();
    $net2->load_parameters('net_fill.params', ctx=>$ctx);
    ok($net2->[0]->weight->shape->[1] == 3);
    ok($net2->[1]->gamma->shape->[0] == 64);
    ok($net2->[2]->weight->shape->[1] == 3072);
}

test_fill_shape_load();

use JSON::PP qw(decode_json);

sub test_inline
{
    my $y;

    my $net = nn->HybridSequential();
    $net->name_scope(sub {
        $net->add(nn->Dense(10));
        $net->add(nn->Dense(10));
        $net->add(nn->Dense(10));
    });
    $net->initialize();

    $net->hybridize(inline_limit=>3);
    mx->autograd->record(sub {
        $y = $net->(mx->nd->zeros([1,10]));
    });
    my $len_1 = @{ decode_json(mx->autograd->get_symbol($y)->tojson())->{nodes} };
    $y->backward();

    $net->hybridize(inline_limit=>0);
    mx->autograd->record(sub {
        $y = $net->(mx->nd->zeros([1,10]));
    });
    my $len_2 = @{ decode_json(mx->autograd->get_symbol($y)->tojson())->{nodes} };
    $y->backward();

    is($len_1, $len_2 + 2);
}

test_inline();

sub test_activations
{
    my $point_to_validate = mx->nd->array([(-0.1, 0.1) x 3]);

    my $swish = nn->Swish();
    my $swish_test = sub { my ($x) = @_;
        return $x * mx->nd->sigmoid($x)
    };

    for(zip($swish_test->($point_to_validate), $swish->($point_to_validate)))
    {
        my ($test_point, $ref_point) = @$_;
        ok($test_point == $ref_point);
    }

    my $elu = nn->ELU();
    my $elu_test = sub { my ($x) = @_;
        my $elu = sub { my ($x) = @_;
            return $x < 0 ? 1.0 * (mx->nd->exp($x) - 1) : $x;
        };
        return [map { $elu->($_) } @{ $x }];
    };

    for(zip($elu_test->($point_to_validate), $elu->($point_to_validate)))
    {
        my ($test_point, $ref_point) = @$_;
        ok($test_point == $ref_point);
    }

    my $selu = nn->SELU();
    my $selu_test = sub { my ($x) = @_;
        my $selu = sub { my ($x) = @_;
            my ($scale, $alpha) = (1.0507009873554804934193349852946, 1.6732632423543772848170429916717);
            return $x => 0 ? $scale * $x : $alpha * mx->nd->exp($x) - $alpha;
        };
        return [map { $selu->($_) } @{ $x }];
    };

    for(zip($selu_test->($point_to_validate), $selu->($point_to_validate)))
    {
        my ($test_point, $ref_point) = @$_;
        ok($test_point == $ref_point);
    }

    my $prelu = nn->PReLU();
    $prelu->initialize();
    my $x = $point_to_validate->reshape([1, 3, 2]);
    ok(almost_equal($prelu->($x)->aspdl, mx->nd->where($x >= 0, $x, 0.25 * $x)->aspdl));
}

test_activations();

sub test_req
{
    my $data = mx->nd->random->uniform(shape=>[1,3,224,224]);
    my $label = mx->nd->array([1]);
    my $loss = gluon->loss->SoftmaxCrossEntropyLoss();

    my $net = nn->HybridSequential();
    my $net1 = nn->HybridSequential();
    $net1->add(nn->Dense(4));
    my $net2 = nn->HybridSequential();
    $net2->add(nn->Dense(3));
    $net2->add(nn->Dense(2));
    $net->add($net1);
    $net->add($net2);
    $net->initialize();

    $net->hybridize();

    for my $v ($net->collect_params->values)
    {
        $v->grad_req('add');
    }

    $net->collect_params->zero_grad();
    my $grad;
    mx->autograd->record(sub {
        my $pred = $net->($data);
        my $l = $loss->($pred, $label);
        $l->backward();
        $grad = $net->[0][0]->weight->grad->mean->aspdl;
        # run twice to check req = add
        $pred = $net->($data);
        $l = $loss->($pred, $label);
        $l->backward;
    });

    my $grad_double = $net->[0][0]->weight->grad->mean->aspdl;
    ok(almost_equal($grad * 2, $grad_double));
}

test_req();

sub test_zero_grad
{
    my $data = mx->nd->random->uniform(shape=>[3,3]);
    my $net = nn->Embedding(3, 4, sparse_grad=>1, prefix=>'test_zero_grad_');
    $net->initialize();
    mx->autograd->record(sub {
        $net->($data)->backward;
    });
    $net->collect_params->zero_grad;
    my $grad = $net->collect_params->params->get('test_zero_grad_weight')->grad;
    ok(almost_equal($grad->aspdl, $grad->aspdl * 0));
}

test_zero_grad();

sub test_hook
{
    my $hook_call_count = 0;
    my $pre_hook_call_count = 0;

    my $call_hook = sub { my ($block, $x, $y) = @_;
        $hook_call_count += 1;
    };

    my $call_pre_hook = sub { my ($block, $x) = @_;
        $pre_hook_call_count += 1;
    };

    my $block = nn->Dense(10);
    $block->initialize();
    my $handle = $block->register_forward_hook($call_hook);
    my $pre_handle = $block->register_forward_pre_hook($call_pre_hook);
    $block->(mx->nd->ones([3, 5]));

    ok($hook_call_count == 1);
    ok($pre_hook_call_count == 1);

    $handle->detach();
    $block->(mx->nd->ones([3, 5]));

    ok($hook_call_count == 1);
    ok($pre_hook_call_count == 2);

    $pre_handle->detach();
    $block->(mx->nd->ones([3, 5]));

    ok($hook_call_count == 1);
    ok($pre_hook_call_count == 2);
}

test_hook();

sub test_apply
{
    my @called_blocks;

    my $record_name = sub { my ($block) = @_;
        push @called_blocks, $block->name;
    };
    my $block = nn->HybridSequential(prefix=>'test_');
    $block->name_scope(sub {
        $block->add(nn->Dense(10));
        $block->add(nn->Dropout(0.5));
    });
    $block->apply($record_name);

    is_deeply(\@called_blocks, ['test_dense0', 'test_dropout0', 'test']);
}

test_apply();

sub test_sparse_hybrid_block_grad
{
    package Embedding {
        use AI::MXNet::Gluon::Mouse;
        use AI::MXNet::Function::Parameters;
        extends 'AI::MXNet::Gluon::HybridBlock';
        has ['num_tokens', 'embedding_size'] => (is => 'rw');
        method python_constructor_arguments() { ['num_tokens', 'embedding_size'] }
        sub BUILD {
            my $self = shift;
            $self->name_scope(sub {
                $self->embedding(nn->Embedding(
                    $self->num_tokens, $self->embedding_size, sparse_grad=>1
                ));
            });
        }

        method hybrid_forward($F, $words)
        {
            my $emb = $self->embedding->($words);
            return $emb + $F->ones_like($emb);
        }
    };
    my $embedding = Embedding->new(20, 3);
    $embedding->initialize();
    $embedding->hybridize();

    my $loss;
    mx->autograd->record(sub {
        my $emb0 = $embedding->(mx->nd->arange(stop => 10))->sum;
        my $emb1 = $embedding->(mx->nd->arange(stop => 10))->sum;
        $loss = $emb0 + $emb1;
    });
    $loss->backward();
    my $grad = $embedding->embedding->weight->grad->aspdl;
    ok(($grad->slice('X', ':9') == 2)->all);
    ok(($grad->slice('X', '10:') == 0)->all);
}

test_sparse_hybrid_block_grad();

sub test_sparse_hybrid_block
{
    package Linear {
        use AI::MXNet::Gluon::Mouse;
        use AI::MXNet::Function::Parameters;
        extends 'AI::MXNet::Gluon::HybridBlock';
        has ['units'] => (is => 'rw');
        method python_constructor_arguments() { ['units'] }
        sub BUILD {
            my $self = shift;
            $self->name_scope(sub {
                $self->w($self->params->get(
                    'w', shape => [$self->units, $self->units]
                ));
            });
        }
        method hybrid_forward($F, $x, :$w)
        {
            return $F->dot($x, $w);
        }
    };
    package SparseBlock {
        use AI::MXNet::Gluon::Mouse;
        use AI::MXNet::Function::Parameters;
        extends 'AI::MXNet::Gluon::HybridBlock';
        has ['units'] => (is => 'rw');
        method python_constructor_arguments() { ['units'] }
        sub BUILD {
            my $self = shift;
            $self->name_scope(sub {
                $self->net(Linear->new($self->units));
            });
        }
        method hybrid_forward($F, $x)
        {
            return $self->net->($x) * $x;
        }
    };
    my $block = SparseBlock->new(2);
    $block->initialize();
    $block->hybridize();
    my $x = mx->nd->ones([2,2])->tostype('csr');
    my $z;
    mx->autograd->record(sub {
        $z = $block->($x) + $block->($x);
    });
    $z->backward;
    ok(($block->net->w->grad->aspdl == 4)->all);
}

test_sparse_hybrid_block();
