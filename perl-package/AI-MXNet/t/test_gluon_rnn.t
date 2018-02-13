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
use Test::More tests => 77;
use AI::MXNet 'mx';
use AI::MXNet::Gluon 'gluon';
use AI::MXNet::TestUtils qw/allclose almost_equal/;
use AI::MXNet::Base;
use Scalar::Util 'blessed';

sub test_rnn
{
    my $cell = gluon->rnn->RNNCell(100, prefix=>'rnn_');
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort $cell->collect_params()->keys()], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);

    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_rnn();

sub test_lstm
{
    my $cell = gluon->rnn->LSTMCell(100, prefix=>'rnn_');
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort $cell->collect_params()->keys()], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);

    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_lstm();

sub test_lstm_forget_bias
{
    my $forget_bias = 2;
    my $stack = gluon->rnn->SequentialRNNCell();
    $stack->add(gluon->rnn->LSTMCell(100, i2h_bias_initializer=>mx->init->LSTMBias($forget_bias), prefix=>'l0_'));
    $stack->add(gluon->rnn->LSTMCell(100, i2h_bias_initializer=>mx->init->LSTMBias($forget_bias), prefix=>'l1_'));

    my $dshape = [32, 1, 200];
    my $data = mx->sym->Variable('data');

    my ($sym) = $stack->unroll(1, $data, merge_outputs=>1);
    my $mod = mx->mod->Module($sym, context=>mx->cpu(0));
    $mod->bind(data_shapes=>[['data', $dshape]]);

    $mod->init_params();

    my ($bias_argument) = grep { /i2h_bias$/ } @{ $sym->list_arguments() };
    my $expected_bias = pdl((0)x100, ($forget_bias)x100, (0)x200);
    ok(allclose(($mod->get_params())[0]->{$bias_argument}->aspdl, $expected_bias));
}

test_lstm_forget_bias();

sub test_gru
{
    my $cell = gluon->rnn->GRUCell(100, prefix=>'rnn_');
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort $cell->collect_params()->keys()], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);

    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_gru();

sub test_residual
{
    my $cell = gluon->rnn->ResidualCell(gluon->rnn->GRUCell(50, prefix=>'rnn_'));
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..1];
    my ($outputs) = $cell->unroll(2, $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort $cell->collect_params()->keys()], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50]);
    is_deeply($outs, [[10, 50], [10, 50]]);
    $outputs = $outputs->eval(args => { rnn_t0_data=>mx->nd->ones([10, 50]),
                           rnn_t1_data=>mx->nd->ones([10, 50]),
                           rnn_i2h_weight=>mx->nd->zeros([150, 50]),
                           rnn_i2h_bias=>mx->nd->zeros([150]),
                           rnn_h2h_weight=>mx->nd->zeros([150, 50]),
                           rnn_h2h_bias=>mx->nd->zeros([150]) });
    my $expected_outputs = mx->nd->ones([10, 50]);
    ok(($outputs->[0] == $expected_outputs)->aspdl->all);
    ok(($outputs->[1] == $expected_outputs)->aspdl->all);
}

test_residual();

sub test_residual_bidirectional
{
    my $cell = gluon->rnn->ResidualCell(
        gluon->rnn->BidirectionalCell(
            gluon->rnn->GRUCell(25, prefix=>'rnn_l_'),
            gluon->rnn->GRUCell(25, prefix=>'rnn_r_')
        )
    );
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..1];
    my ($outputs) = $cell->unroll(2, $inputs, merge_outputs => 0);
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort $cell->collect_params()->keys()],
                ['rnn_l_h2h_bias', 'rnn_l_h2h_weight', 'rnn_l_i2h_bias', 'rnn_l_i2h_weight',
                'rnn_r_h2h_bias', 'rnn_r_h2h_weight', 'rnn_r_i2h_bias', 'rnn_r_i2h_weight']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50]);
    is_deeply($outs, [[10, 50], [10, 50]]);
    $outputs = $outputs->eval(args => { rnn_t0_data=>mx->nd->ones([10, 50])+5,
                           rnn_t1_data=>mx->nd->ones([10, 50])+5,
                           rnn_l_i2h_weight=>mx->nd->zeros([75, 50]),
                           rnn_l_i2h_bias=>mx->nd->zeros([75]),
                           rnn_l_h2h_weight=>mx->nd->zeros([75, 25]),
                           rnn_l_h2h_bias=>mx->nd->zeros([75]),
                           rnn_r_i2h_weight=>mx->nd->zeros([75, 50]),
                           rnn_r_i2h_bias=>mx->nd->zeros([75]),
                           rnn_r_h2h_weight=>mx->nd->zeros([75, 25]),
                           rnn_r_h2h_bias=>mx->nd->zeros([75]),
    });
    my $expected_outputs = mx->nd->ones([10, 50])+5;
    ok(($outputs->[0] == $expected_outputs)->aspdl->all);
    ok(($outputs->[1] == $expected_outputs)->aspdl->all);
}

test_residual_bidirectional();

sub test_stack
{
    my $cell = gluon->rnn->SequentialRNNCell();
    for my $i (0..4)
    {
        if($i == 1)
        {
            $cell->add(gluon->rnn->ResidualCell(gluon->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_")));
        }
        else
        {
            $cell->add(gluon->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_"));
        }
    }
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    my %keys = map { $_ => 1 } $cell->collect_params()->keys();
    for my $i (0..4)
    {
        ok($keys{"rnn_stack${i}_h2h_weight"});
        ok($keys{"rnn_stack${i}_h2h_bias"});
        ok($keys{"rnn_stack${i}_i2h_weight"});
        ok($keys{"rnn_stack${i}_i2h_bias"});
    }
    is_deeply($outputs->list_outputs(), ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_stack();

sub test_bidirectional
{
    my $cell = gluon->rnn->BidirectionalCell(
            gluon->rnn->LSTMCell(100, prefix=>'rnn_l0_'),
            gluon->rnn->LSTMCell(100, prefix=>'rnn_r0_'),
            output_prefix=>'rnn_bi_');
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply($outputs->list_outputs(), ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 200], [10, 200], [10, 200]]);
}

test_bidirectional();

sub test_zoneout
{
    my $cell = gluon->rnn->ZoneoutCell(gluon->rnn->RNNCell(100, prefix=>'rnn_'), zoneout_outputs=>0.5,
                              zoneout_states=>0.5);
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, $inputs);
    $outputs = mx->sym->Group($outputs);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_zoneout();

sub check_rnn_forward
{
    my ($layer, $inputs, $deterministic) = @_;
    $deterministic //= 1;
    $inputs->attach_grad();
    $layer->collect_params()->initialize();
    my $out;
    mx->autograd->record(sub {
        $out = ($layer->unroll(3, $inputs, merge_outputs=>0))[0];
        mx->autograd->backward($out);
        $out = ($layer->unroll(3, $inputs, merge_outputs=>1))[0];
        $out->backward;
    });

    my $pdl_out = $out->aspdl;
    my $pdl_dx = $inputs->grad->aspdl;

    $layer->hybridize;

    mx->autograd->record(sub {
        $out = ($layer->unroll(3, $inputs, merge_outputs=>0))[0];
        mx->autograd->backward($out);
        $out = ($layer->unroll(3, $inputs, merge_outputs=>1))[0];
        $out->backward;
    });

    if($deterministic)
    {
        ok(almost_equal($pdl_out, $out->aspdl, 1e-3));
        ok(almost_equal($pdl_dx, $inputs->grad->aspdl, 1e-3));
    }
}

sub test_rnn_cells
{
    check_rnn_forward(gluon->rnn->LSTMCell(100, input_size=>200), mx->nd->ones([8, 3, 200]));
    check_rnn_forward(gluon->rnn->RNNCell(100, input_size=>200), mx->nd->ones([8, 3, 200]));
    check_rnn_forward(gluon->rnn->GRUCell(100, input_size=>200), mx->nd->ones([8, 3, 200]));
    my $bilayer = gluon->rnn->BidirectionalCell(
        gluon->rnn->LSTMCell(100, input_size=>200),
        gluon->rnn->LSTMCell(100, input_size=>200)
    );
    check_rnn_forward($bilayer, mx->nd->ones([8, 3, 200]));
    check_rnn_forward(gluon->rnn->DropoutCell(0.5), mx->nd->ones([8, 3, 200]), 0);
    check_rnn_forward(
        gluon->rnn->ZoneoutCell(
            gluon->rnn->LSTMCell(100, input_size=>200),
            0.5, 0.2
        ),
        mx->nd->ones([8, 3, 200]),
        0
    );
    my $net = gluon->rnn->SequentialRNNCell();
    $net->add(gluon->rnn->LSTMCell(100, input_size=>200));
    $net->add(gluon->rnn->RNNCell(100, input_size=>100));
    $net->add(gluon->rnn->GRUCell(100, input_size=>100));
    check_rnn_forward($net, mx->nd->ones([8, 3, 200]));
}

test_rnn_cells();

sub check_rnn_layer_forward
{
    my ($layer, $inputs, $states) = @_;
    $layer->collect_params()->initialize();
    $inputs->attach_grad;
    my $out;
    mx->autograd->record(sub {
        $out = $layer->($inputs, $states);
        if(defined $states)
        {
            ok(@$out == 2);
            $out = $out->[0];
        }
        else
        {
            ok(blessed $out and $out->isa('AI::MXNet::NDArray'));
        }
        $out->backward();
    });

    my $pdl_out = $out->aspdl;
    my $pdl_dx = $inputs->grad->aspdl;
    $layer->hybridize;

    mx->autograd->record(sub {
        $out = $layer->($inputs, $states);
        if(defined $states)
        {
            ok(@$out == 2);
            $out = $out->[0]
        }
        else
        {
            ok(blessed $out and $out->isa('AI::MXNet::NDArray'));
        }
        $out->backward();
    });

    ok(almost_equal($pdl_out, $out->aspdl, 1e-3));
    ok(almost_equal($pdl_dx, $inputs->grad->aspdl, 1e-3));
}

sub test_rnn_layers
{
    check_rnn_layer_forward(gluon->rnn->RNN(10, 2), mx->nd->ones([8, 3, 20]));
    check_rnn_layer_forward(gluon->rnn->RNN(10, 2), mx->nd->ones([8, 3, 20]), mx->nd->ones([2, 3, 10]));
    check_rnn_layer_forward(gluon->rnn->LSTM(10, 2), mx->nd->ones([8, 3, 20]));
    check_rnn_layer_forward(gluon->rnn->LSTM(10, 2), mx->nd->ones([8, 3, 20]), [mx->nd->ones([2, 3, 10]), mx->nd->ones([2, 3, 10])]);
    check_rnn_layer_forward(gluon->rnn->GRU(10, 2), mx->nd->ones([8, 3, 20]));
    check_rnn_layer_forward(gluon->rnn->GRU(10, 2), mx->nd->ones([8, 3, 20]), mx->nd->ones([2, 3, 10]));

#    my $net = gluon->nn->Sequential();
#    $net->add(gluon->rnn->LSTM(10, 2, bidirectional=>1));
#    $net->add(gluon->nn->BatchNorm(axis=>2));
#    $net->add(gluon->nn->Flatten());
#    $net->add(gluon->nn->Dense(3, activation=>'relu'));
#    $net->collect_params()->initialize();
#    mx->autograd->record(sub {
#        $net->(mx->nd->ones([2, 3, 10]))->backward();
#    });
}

test_rnn_layers();
