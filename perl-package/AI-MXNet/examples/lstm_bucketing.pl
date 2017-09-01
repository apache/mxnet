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
use PDL;
use AI::MXNet qw(mx);
use AI::MXNet::Function::Parameters;
use Getopt::Long qw(HelpMessage);

GetOptions(
    'num-layers=i'   => \(my $num_layers   = 2       ),
    'num-hidden=i'   => \(my $num_hidden   = 200     ),
    'num-embed=i'    => \(my $num_embed    = 200     ),
    'gpus=s'         => \(my $gpus                   ),
    'kv-store=s'     => \(my $kv_store     = 'device'),
    'num-epoch=i'    => \(my $num_epoch    = 25      ),
    'lr=f'           => \(my $lr           = 0.01    ),
    'optimizer=s'    => \(my $optimizer    = 'sgd'   ),
    'mom=f'          => \(my $mom          = 0       ),
    'wd=f'           => \(my $wd           = 0.00001 ),
    'batch-size=i'   => \(my $batch_size   = 32      ),
    'disp-batches=i' => \(my $disp_batches = 50      ),
    'chkp-prefix=s'  => \(my $chkp_prefix  = 'lstm_' ),
    'chkp-epoch=i'   => \(my $chkp_epoch   = 0       ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    lstm_bucketing.pl - Example of training LSTM RNN on Penn Tree Bank data using high level RNN interface

=head1 SYNOPSIS

    --num-layers     number of stacked RNN layers, default=2
    --num-hidden     hidden layer size, default=200
    --num-embed      embedding layer size, default=200
    --gpus           list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.
                     Increase batch size when using multiple gpus for best performance.
    --kv-store       key-value store type, default='device'
    --num-epochs     max num of epochs, default=25
    --lr             initial learning rate, default=0.01
    --optimizer      the optimizer type, default='sgd'
    --mom            momentum for sgd, default=0.0
    --wd             weight decay for sgd, default=0.00001
    --batch-size     the batch size type, default=32
    --disp-batches   show progress for every n batches, default=50
    --chkp-prefix    prefix for checkpoint files, default='lstm_'
    --chkp-epoch     save checkpoint after this many epoch, default=0 (saving checkpoints is disabled)

=cut
func tokenize_text($fname, :$vocab=, :$invalid_label=-1, :$start_label=0)
{
    open(F, $fname) or die "Can't open $fname: $!";
    my @lines = map { my $l = [split(/ /)]; shift(@$l); $l } (<F>);
    my $sentences;
    ($sentences, $vocab) = mx->rnn->encode_sentences(
        \@lines,
        vocab         => $vocab,
        invalid_label => $invalid_label,
        start_label   => $start_label
    );
    return ($sentences, $vocab);
}

my $buckets = [10, 20, 30, 40, 50, 60];
my $start_label   = 1;
my $invalid_label = 0;

my ($train_sentences, $vocabulary) = tokenize_text(
    './data/ptb.train.txt', start_label => $start_label,
    invalid_label => $invalid_label
);
my ($validation_sentences) = tokenize_text(
    './data/ptb.test.txt', vocab => $vocabulary,
    start_label => $start_label, invalid_label => $invalid_label
);
my $data_train  = mx->rnn->BucketSentenceIter(
    $train_sentences, $batch_size, buckets => $buckets,
    invalid_label => $invalid_label
);
my $data_val    = mx->rnn->BucketSentenceIter(
    $validation_sentences, $batch_size, buckets => $buckets,
    invalid_label => $invalid_label
);

my $stack = mx->rnn->SequentialRNNCell();
for my $i (0..$num_layers-1)
{
    $stack->add(mx->rnn->LSTMCell(num_hidden => $num_hidden, prefix => "lstm_l${i}_"));
}

my $sym_gen = sub {
    my $seq_len = shift;
    my $data  = mx->sym->Variable('data');
    my $label = mx->sym->Variable('softmax_label');
    my $embed = mx->sym->Embedding(
        data => $data, input_dim => scalar(keys %$vocabulary),
        output_dim => $num_embed, name => 'embed'
    );
    $stack->reset;
    my ($outputs, $states) = $stack->unroll($seq_len, inputs => $embed, merge_outputs => 1);
    my $pred = mx->sym->Reshape($outputs, shape => [-1, $num_hidden]);
    $pred    = mx->sym->FullyConnected(data => $pred, num_hidden => scalar(keys %$vocabulary), name => 'pred');
    $label   = mx->sym->Reshape($label, shape => [-1]);
    $pred    = mx->sym->SoftmaxOutput(data => $pred, label => $label, name => 'softmax');
    return ($pred, ['data'], ['softmax_label']);
};

my $contexts;
if(defined $gpus)
{
    $contexts = [map { mx->gpu($_) } split(/,/, $gpus)];
}
else
{
    $contexts = mx->cpu(0);
}

my $model = mx->mod->BucketingModule(
    sym_gen             => $sym_gen,
    default_bucket_key  => $data_train->default_bucket_key,
    context             => $contexts
);

$model->fit(
    $data_train,
    eval_data           => $data_val,
    eval_metric         => mx->metric->Perplexity($invalid_label),
    kvstore             => $kv_store,
    optimizer           => $optimizer,
    optimizer_params    => {
                                learning_rate => $lr,
                                momentum      => $mom,
                                wd            => $wd,
                        },
    initializer         => mx->init->Xavier(factor_type => "in", magnitude => 2.34),
    num_epoch           => $num_epoch,
    batch_end_callback  => mx->callback->Speedometer($batch_size, $disp_batches),
    ($chkp_epoch ? (epoch_end_callback  => mx->rnn->do_rnn_checkpoint($stack, $chkp_prefix, $chkp_epoch)) : ())
);
