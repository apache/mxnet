use strict;
use warnings;
use AI::MXNet qw(mx);
use PDL;
use Test::More tests => 28;
use Data::Dumper;

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
    my $cell = mx->rnn->LSTMCell(100, prefix=>'rnn_');
    my($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_stack
{
    my $cell = mx->rnn->SequentialRNNCell();
    for my $i (0..4)
    {
        $cell->add(mx->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_"));
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

test_rnn();
test_lstm();
test_stack();