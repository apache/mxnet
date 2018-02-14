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
use AI::MXNet::TestUtils qw(same);
use PDL;
use Test::More tests => 54;

sub test_rnn
{
    my $cell = mx->rnn->RNNCell(100, prefix=>'rnn_');
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_lstm
{
    my $cell = mx->rnn->LSTMCell(100, prefix=>'rnn_', forget_bias => 1);
    my($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_lstm_forget_bias
{
    my $forget_bias = 2;
    my $stack = mx->rnn->SequentialRNNCell();
    $stack->add(mx->rnn->LSTMCell(100, forget_bias=>$forget_bias, prefix=>'l0_'));
    $stack->add(mx->rnn->LSTMCell(100, forget_bias=>$forget_bias, prefix=>'l1_'));

    my $dshape = [32, 1, 200];
    my $data   = mx->sym->Variable('data');

    my ($sym) = $stack->unroll(1, inputs => $data, merge_outputs => 1);
    my $mod = mx->mod->Module($sym, context => mx->cpu(0));
    $mod->bind(data_shapes=>[['data', $dshape]]);

    $mod->init_params();
    my ($bias_argument) = grep { /i2h_bias$/ } @{ $sym->list_arguments };
    my $f = zeros(100);
    my $expected_bias = $f->glue(0, $forget_bias * ones(100), zeros(200));
    ok(
        ((($mod->get_params())[0]->{$bias_argument}->aspdl - $expected_bias)->abs < 1e-07)->all
    );
}

sub test_gru
{
    my $cell = mx->rnn->GRUCell(100, prefix=>'rnn_');
    my($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_residual
{
    my $cell = mx->rnn->ResidualCell(mx->rnn->GRUCell(50, prefix=>'rnn_'));
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..1];
    my ($outputs)= $cell->unroll(2, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply(
        [sort keys %{ $cell->params->_params }],
        ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    );
    is_deeply(
        $outputs->list_outputs,
        ['rnn_t0_out_plus_residual_output', 'rnn_t1_out_plus_residual_output']
    );

    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10, 50], rnn_t1_data=>[10, 50]);
    is_deeply($outs, [[10, 50], [10, 50]]);
    $outputs = $outputs->eval(args => {
        rnn_t0_data=>mx->nd->ones([10, 50]),
        rnn_t1_data=>mx->nd->ones([10, 50]),
        rnn_i2h_weight=>mx->nd->zeros([150, 50]),
        rnn_i2h_bias=>mx->nd->zeros([150]),
        rnn_h2h_weight=>mx->nd->zeros([150, 50]),
        rnn_h2h_bias=>mx->nd->zeros([150])
    });
    my $expected_outputs = mx->nd->ones([10, 50])->aspdl;
    same(@{$outputs}[0]->aspdl, $expected_outputs);
    same(@{$outputs}[1]->aspdl, $expected_outputs);
}

sub test_residual_bidirectional
{
    my $cell = mx->rnn->ResidualCell(
        mx->rnn->BidirectionalCell(
            mx->rnn->GRUCell(25, prefix=>'rnn_l_'),
            mx->rnn->GRUCell(25, prefix=>'rnn_r_')
        )
    );
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..1];
    my ($outputs) = $cell->unroll(2, inputs => $inputs, merge_outputs=>0);
    $outputs = mx->sym->Group($outputs);
    is_deeply(
        [sort keys %{ $cell->params->_params }],
        ['rnn_l_h2h_bias', 'rnn_l_h2h_weight', 'rnn_l_i2h_bias', 'rnn_l_i2h_weight',
        'rnn_r_h2h_bias', 'rnn_r_h2h_weight', 'rnn_r_i2h_bias', 'rnn_r_i2h_weight']
    );
    is_deeply(
        $outputs->list_outputs,
        ['bi_t0_plus_residual_output', 'bi_t1_plus_residual_output']
    );

    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10, 50], rnn_t1_data=>[10, 50]);
    is_deeply($outs, [[10, 50], [10, 50]]);
    $outputs = $outputs->eval(args => {
        rnn_t0_data=>mx->nd->ones([10, 50])+5,
        rnn_t1_data=>mx->nd->ones([10, 50])+5,
        rnn_l_i2h_weight=>mx->nd->zeros([75, 50]),
        rnn_l_i2h_bias=>mx->nd->zeros([75]),
        rnn_l_h2h_weight=>mx->nd->zeros([75, 25]),
        rnn_l_h2h_bias=>mx->nd->zeros([75]),
        rnn_r_i2h_weight=>mx->nd->zeros([75, 50]),
        rnn_r_i2h_bias=>mx->nd->zeros([75]),
        rnn_r_h2h_weight=>mx->nd->zeros([75, 25]),
        rnn_r_h2h_bias=>mx->nd->zeros([75])
    });
    my $expected_outputs = (mx->nd->ones([10, 50])+5)->aspdl;
    ok(same(@{$outputs}[0]->aspdl, $expected_outputs));
    ok(same(@{$outputs}[1]->aspdl, $expected_outputs));
}

sub test_stack
{
    my $cell = mx->rnn->SequentialRNNCell();
    for my $i (0..4)
    {
        if($i == 1)
        {
            $cell->add(mx->rnn->ResidualCell(mx->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_")));
        }
        else
        {
            $cell->add(mx->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_"));
        }
    }
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    my %params = %{ $cell->params->_params };
    for my $i (0..4)
    {
        ok(exists $params{"rnn_stack${i}_h2h_weight"});
        ok(exists $params{"rnn_stack${i}_h2h_bias"});
        ok(exists $params{"rnn_stack${i}_i2h_weight"});
        ok(exists $params{"rnn_stack${i}_i2h_bias"});
    }
    is_deeply($outputs->list_outputs(), ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_bidirectional
{
    my $cell = mx->rnn->BidirectionalCell(
        mx->rnn->LSTMCell(100, prefix=>'rnn_l0_'),
        mx->rnn->LSTMCell(100, prefix=>'rnn_r0_'),
        output_prefix=>'rnn_bi_'
    );
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply($outputs->list_outputs(), ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 200], [10, 200], [10, 200]]);
}

sub test_unfuse
{
    my $cell = mx->rnn->FusedRNNCell(
        100, num_layers => 1, mode => 'lstm',
        prefix => 'test_', bidirectional => 1
    )->unfuse;
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply($outputs->list_outputs(), ['test_bi_lstm_0t0_output', 'test_bi_lstm_0t1_output', 'test_bi_lstm_0t2_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 200], [10, 200], [10, 200]]);
}

sub test_zoneout
{
    my $cell = mx->rnn->ZoneoutCell(
        mx->rnn->RNNCell(100, prefix=>'rnn_'),
        zoneout_outputs => 0.5,
        zoneout_states  => 0.5
    );
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10, 50], rnn_t1_data=>[10, 50], rnn_t2_data=>[10, 50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_convrnn
{
    my $cell = mx->rnn->ConvRNNCell(input_shape => [1, 3, 16, 10], num_hidden=>10,
                              h2h_kernel=>[3, 3], h2h_dilate=>[1, 1],
                              i2h_kernel=>[3, 3], i2h_stride=>[1, 1],
                              i2h_pad=>[1, 1], i2h_dilate=>[1, 1],
                              prefix=>'rnn_');
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply(
        [sort keys %{ $cell->params->_params }],
        ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    );
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[1, 3, 16, 10], rnn_t1_data=>[1, 3, 16, 10], rnn_t2_data=>[1, 3, 16, 10]);
    is_deeply($outs, [[1, 10, 16, 10], [1, 10, 16, 10], [1, 10, 16, 10]]);
}

sub test_convlstm
{
    my $cell = mx->rnn->ConvLSTMCell(input_shape => [1, 3, 16, 10], num_hidden=>10,
                              h2h_kernel=>[3, 3], h2h_dilate=>[1, 1],
                              i2h_kernel=>[3, 3], i2h_stride=>[1, 1],
                              i2h_pad=>[1, 1], i2h_dilate=>[1, 1],
                              prefix=>'rnn_', forget_bias => 1);
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply(
        [sort keys %{ $cell->params->_params }],
        ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    );
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[1, 3, 16, 10], rnn_t1_data=>[1, 3, 16, 10], rnn_t2_data=>[1, 3, 16, 10]);
    is_deeply($outs, [[1, 10, 16, 10], [1, 10, 16, 10], [1, 10, 16, 10]]);
}

sub test_convgru
{
    my $cell = mx->rnn->ConvGRUCell(input_shape => [1, 3, 16, 10], num_hidden=>10,
                              h2h_kernel=>[3, 3], h2h_dilate=>[1, 1],
                              i2h_kernel=>[3, 3], i2h_stride=>[1, 1],
                              i2h_pad=>[1, 1], i2h_dilate=>[1, 1],
                              prefix=>'rnn_', forget_bias => 1);
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    is_deeply(
        [sort keys %{ $cell->params->_params }],
        ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']
    );
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[1, 3, 16, 10], rnn_t1_data=>[1, 3, 16, 10], rnn_t2_data=>[1, 3, 16, 10]);
    is_deeply($outs, [[1, 10, 16, 10], [1, 10, 16, 10], [1, 10, 16, 10]]);
}

test_rnn();
test_lstm();
test_lstm_forget_bias();
test_gru();
test_residual();
test_residual_bidirectional();
test_stack();
test_bidirectional();
test_unfuse();
test_zoneout();
test_convrnn();
test_convlstm();
test_convgru();
