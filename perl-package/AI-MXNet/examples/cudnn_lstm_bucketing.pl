#!/usr/bin/perl
use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;
use Getopt::Long qw(HelpMessage);

GetOptions(
    'test'            => \(my $do_test                ),
    'num-layers=i'    => \(my $num_layers   = 2       ),
    'num-hidden=i'    => \(my $num_hidden   = 256     ),
    'num-embed=i'     => \(my $num_embed    = 256     ),
    'num-seq=i'       => \(my $seq_size     = 32      ),
    'gpus=s'          => \(my $gpus                   ),
    'kv-store=s'      => \(my $kv_store     = 'device'),
    'num-epoch=i'     => \(my $num_epoch    = 25      ),
    'lr=f'            => \(my $lr           = 0.01    ),
    'optimizer=s'     => \(my $optimizer    = 'adam'  ),
    'mom=f'           => \(my $mom          = 0       ),
    'wd=f'            => \(my $wd           = 0.00001 ),
    'batch-size=i'    => \(my $batch_size   = 32      ),
    'disp-batches=i'  => \(my $disp_batches = 50      ),
    'model-prefix=s'  => \(my $model_prefix = 'lstm_' ),
    'load-epoch=i'    => \(my $load_epoch   = 0       ),
    'stack-rnn'       => \(my $stack_rnn              ),
    'bidirectional=i' => \(my $bidirectional          ),
    'dropout=f',      => \(my $dropout      = 0       ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    char_lstm.pl - Example of training char LSTM RNN on tiny shakespeare using high level RNN interface

=head1 SYNOPSIS

    --test           Whether to test or train (default 0)
    --num-layers     number of stacked RNN layers, default=2
    --num-hidden     hidden layer size, default=200
    --num-seq        sequence size, default=32
    --gpus           list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.
                     Increase batch size when using multiple gpus for best performance.
    --kv-store       key-value store type, default='device'
    --num-epochs     max num of epochs, default=25
    --lr             initial learning rate, default=0.01
    --optimizer      the optimizer type, default='adam'
    --mom            momentum for sgd, default=0.0
    --wd             weight decay for sgd, default=0.00001
    --batch-size     the batch size type, default=32
    --disp-batches   show progress for every n batches, default=50
    --model-prefix   prefix for checkpoint files for loading/saving, default='lstm_'
    --load-epoch     load from epoch
    --stack-rnn      stack rnn to reduce communication overhead (1,0 default 0)
    --bidirectional  whether to use bidirectional layers (1,0 default 0)
    --dropout        dropout probability (1.0 - keep probability), default 0
=cut

$bidirectional = $bidirectional ? 1 : 0;
$stack_rnn     = $stack_rnn     ? 1 : 0;

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

func get_data($layout)
{
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
        invalid_label => $invalid_label,
        layout        => $layout
    );
    my $data_val    = mx->rnn->BucketSentenceIter(
        $validation_sentences, $batch_size, buckets => $buckets,
        invalid_label => $invalid_label,
        layout        => $layout
    );
    return ($data_train, $data_val, $vocabulary);
}

my $train = sub
{
    my ($data_train, $data_val, $vocab) = get_data('TN');
    my $cell;
    if($stack_rnn)
    {
        my $stack = mx->rnn->SequentialRNNCell();
        for my $i (0..$num_layers-1)
        {
            my $dropout_rate = 0;
            if($i < $num_layers-1)
            {
                $dropout_rate = $dropout;
            }
            $stack->add(
                mx->rnn->FusedRNNCell(
                    $num_hidden, num_layers => 1,
                    mode => 'lstm', prefix => "lstm_$i",
                    bidirectional => $bidirectional, dropout => $dropout_rate
                )
            );
        }
        $cell = $stack;
    }
    else
    {
        $cell = mx->rnn->FusedRNNCell(
            $num_hidden, mode => 'lstm', num_layers => $num_layers,
            bidirectional => $bidirectional, dropout => $dropout
        );
    }

    my $sym_gen = sub { my $seq_len = shift;
        my $data = mx->sym->Variable('data');
        my $label = mx->sym->Variable('softmax_label');
        my $embed = mx->sym->Embedding(data=>$data, input_dim=>scalar(keys %$vocab), output_dim=>$num_embed,name=>'embed');
        my ($output) = $cell->unroll($seq_len, inputs=>$embed, merge_outputs=>1, layout=>'TNC');
        my $pred = mx->sym->Reshape($output, shape=>[-1, $num_hidden*(1+$bidirectional)]);
        $pred = mx->sym->FullyConnected(data=>$pred, num_hidden=>scalar(keys %$vocab), name=>'pred');
        $label = mx->sym->Reshape($label, shape=>[-1]);
        $pred = mx->sym->SoftmaxOutput(data=>$pred, label=>$label, name=>'softmax');
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

    my ($arg_params, $aux_params);
    if($load_epoch)
    {
        (undef, $arg_params, $aux_params) = mx->rnn->load_rnn_checkpoint(
            $cell, $model_prefix, $load_epoch);
    }
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
        begin_epoch         => $load_epoch,
        initializer         => mx->init->Xavier(factor_type => "in", magnitude => 2.34),
        num_epoch           => $num_epoch,
        batch_end_callback  => mx->callback->Speedometer($batch_size, $disp_batches),
        ($model_prefix ? (epoch_end_callback  => mx->rnn->do_rnn_checkpoint($cell, $model_prefix, 1)) : ())
    );
};

my $test = sub {
    assert($model_prefix, "Must specifiy path to load from");
    my (undef, $data_val, $vocab) = get_data('NT');
    my $stack;
    if($stack_rnn)
    {
        $stack = mx->rnn->SequentialRNNCell();
        for my $i (0..$num_layers-1)
        {
            my $cell = mx->rnn->LSTMCell(num_hidden => $num_hidden, prefix => "lstm_${i}l0_");
            if($bidirectional)
            {
                $cell = mx->rnn->BidirectionalCell(
                    $cell,
                    mx->rnn->LSTMCell(
                        num_hidden => $num_hidden,
                        prefix => "lstm_${i}r0_"
                    ),
                    output_prefix => "bi_lstm_$i"
                );
            }
            $stack->add($cell);
        }
    }
    else
    {
        $stack = mx->rnn->FusedRNNCell(
            $num_hidden,  num_layers    => $num_layers,
            mode=>'lstm', bidirectional => $bidirectional
        )->unfuse()
    }
    my $sym_gen = sub {
        my $seq_len = shift;
        my $data  = mx->sym->Variable('data');
        my $label = mx->sym->Variable('softmax_label');
        my $embed = mx->sym->Embedding(
            data => $data, input_dim => scalar(keys %$vocab),
            output_dim => $num_embed, name => 'embed'
        );
        $stack->reset;
        my ($outputs, $states) = $stack->unroll($seq_len, inputs => $embed, merge_outputs => 1);
        my $pred = mx->sym->Reshape($outputs, shape => [-1, $num_hidden*(1+$bidirectional)]);
        $pred    = mx->sym->FullyConnected(data => $pred, num_hidden => scalar(keys %$vocab), name => 'pred');
        $label   = mx->sym->Reshape($label, shape => [-1]);
        $pred    = mx->sym->SoftmaxOutput(data => $pred, label => $label, name => 'softmax');
        return ($pred, ['data'], ['softmax_label']);
    };
    my $contexts;
    if($gpus)
    {
        $contexts = [map { mx->gpu($_) } split(/,/, $gpus)];
    }
    else
    {
        $contexts = mx->cpu(0);
    }

    my ($arg_params, $aux_params);
    if($load_epoch)
    {
        (undef, $arg_params, $aux_params) = mx->rnn->load_rnn_checkpoint(
            $stack, $model_prefix, $load_epoch);
    }
    my $model = mx->mod->BucketingModule(
        sym_gen             => $sym_gen,
        default_bucket_key  => $data_val->default_bucket_key,
        context             => $contexts
    );
    $model->bind(
        data_shapes  => $data_val->provide_data,
        label_shapes => $data_val->provide_label,
        for_training => 0,
        force_rebind => 0
    );
    $model->set_params($arg_params, $aux_params);
    my $score = $model->score($data_val,
        mx->metric->Perplexity($invalid_label),
        batch_end_callback=>mx->callback->Speedometer($batch_size, 5)
    );
};

if($num_layers >= 4 and split(/,/,$gpus) >= 4 and not $stack_rnn)
{
    print("WARNING: stack-rnn is recommended to train complex model on multiple GPUs\n");
}

if($do_test)
{
    # Demonstrates how to load a model trained with CuDNN RNN and predict
    # with non-fused MXNet symbol
    $test->();
}
else
{
    $train->();
}