#!/usr/bin/perl
use strict;
use warnings;
use PDL;
use Math::Random::Discrete;
use AI::MXNet qw(mx);
use AI::MXNet::Function::Parameters;
use Getopt::Long qw(HelpMessage);

GetOptions(
    'num-layers=i'   => \(my $num_layers   = 2       ),
    'num-hidden=i'   => \(my $num_hidden   = 256     ),
    'num-embed=i'    => \(my $num_embed    = 10      ),
    'num-seq=i'      => \(my $seq_size     = 60      ),
    'gpus=s'         => \(my $gpus                   ),
    'kv-store=s'     => \(my $kv_store     = 'device'),
    'num-epoch=i'    => \(my $num_epoch    = 25      ),
    'lr=f'           => \(my $lr           = 0.001    ),
    'optimizer=s'    => \(my $optimizer    = 'adam'   ),
    'mom=f'          => \(my $mom          = 0       ),
    'wd=f'           => \(my $wd           = 0.00001 ),
    'batch-size=i'   => \(my $batch_size   = 32      ),
    'disp-batches=i' => \(my $disp_batches = 50      ),
    'chkp-prefix=s'  => \(my $chkp_prefix  = 'lstm_' ),
    'cell-mode=s'    => \(my $cell_mode    = 'LSTM'  ),
    'sample-size=i'  => \(my $sample_size  = 10000   ),
    'chkp-epoch=i'   => \(my $chkp_epoch   = 1       ),
    'bidirectional=i'=> \(my $bidirectional= 0       ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    char_lstm.pl - Example of training char LSTM RNN on tiny shakespeare using high level RNN interface
                   with optional inferred sampling (RNN generates Shakespeare like text)

=head1 SYNOPSIS

    --num-layers     number of stacked RNN layers, default=2
    --num-hidden     hidden layer size, default=256
    --num-embed      embed size, default=10
    --num-seq        sequence size, default=60
    --gpus           list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.
                     Increase batch size when using multiple gpus for best performance.
    --kv-store       key-value store type, default='device'
    --num-epochs     max num of epochs, default=25
    --lr             initial learning rate, default=0.01
    --optimizer      the optimizer type, default='adam'
    --mom            momentum for sgd, default=0.0
    --wd             weight decay for sgd, default=0.00001
    --batch-size     the batch size type, default=32
    --bidirectional  use bidirectional cell, default false (0)
    --disp-batches   show progress for every n batches, default=50
    --chkp-prefix    prefix for checkpoint files, default='lstm_'
    --cell-mode      RNN cell mode (LSTM, GRU, RNN, default=LSTM)
    --sample-size    a size of inferred sample text (default=10000) after each epoch
    --chkp-epoch     save checkpoint after this many epoch, default=1 (saving every checkpoint)

=cut

package AI::MXNet::RNN::IO::ASCIIIterator;
use Mouse;
extends AI::MXNet::DataIter;
has 'data'          => (is => 'ro',  isa => 'PDL',   required => 1);
has 'seq_size'      => (is => 'ro',  isa => 'Int',   required => 1);
has '+batch_size'   => (is => 'ro',  isa => 'Int',   required => 1);
has 'data_name'     => (is => 'ro',  isa => 'Str',   default => 'data');
has 'label_name'    => (is => 'ro',  isa => 'Str',   default => 'softmax_label');
has 'dtype'         => (is => 'ro',  isa => 'Dtype', default => 'float32');
has [qw/nd counter seq_counter vocab_size
    data_size provide_data provide_label idx/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->data_size($self->data->nelem);
    my $segments = int(($self->data_size-$self->seq_size)/($self->batch_size*$self->seq_size));
    $self->idx([0..$segments-1]);
    $self->vocab_size($self->data->uniq->shape->at(0));
    $self->counter(0);
    $self->seq_counter(0);
    $self->nd(mx->nd->array($self->data, dtype => $self->dtype));
    my $shape = [$self->batch_size, $self->seq_size];
    $self->provide_data([
        AI::MXNet::DataDesc->new(
            name  => $self->data_name,
            shape => $shape,
            dtype => $self->dtype
        )
    ]);
    $self->provide_label([
        AI::MXNet::DataDesc->new(
            name  => $self->label_name,
            shape => $shape,
            dtype => $self->dtype
        )
    ]);
    $self->reset;
}

method reset()
{
    $self->counter(0);
    @{ $self->idx } = List::Util::shuffle(@{ $self->idx });
}

method next()
{
    return undef if $self->counter == @{$self->idx};
    my $offset = $self->idx->[$self->counter]*$self->batch_size*$self->seq_size + $self->seq_counter;
    my $data = $self->nd->slice(
        [$offset, $offset + $self->batch_size*$self->seq_size-1]
    )->reshape([$self->batch_size, $self->seq_size]);
    my $label = $self->nd->slice(
        [$offset + 1 , $offset + $self->batch_size*$self->seq_size]
    )->reshape([$self->batch_size, $self->seq_size]);
    $self->seq_counter($self->seq_counter + 1);
    if($self->seq_counter == $seq_size - 1)
    {
        $self->counter($self->counter + 1);
        $self->seq_counter(0);
    }
    return AI::MXNet::DataBatch->new(
        data          => [$data],
        label         => [$label],
        provide_data  => [
            AI::MXNet::DataDesc->new(
                name  => $self->data_name,
                shape => $data->shape,
                dtype => $self->dtype
            )
        ],
        provide_label => [
            AI::MXNet::DataDesc->new(
                name  => $self->label_name,
                shape => $label->shape,
                dtype => $self->dtype
            )
        ],
    );
}

package main;
my $file = "data/input.txt";
open(F, $file) or die "can't open $file: $!";
my $fdata;
{ local($/) = undef; $fdata = <F>; close(F) };
my %vocabulary; my $i = 0;
$fdata = pdl(map{ exists $vocabulary{$_} ? $vocabulary{$_} : ($vocabulary{$_} = $i++) } split(//, $fdata));
my $data_iter = AI::MXNet::RNN::IO::ASCIIIterator->new(
    batch_size => $batch_size,
    data       => $fdata,
    seq_size   => $seq_size
);
my %reverse_vocab = reverse %vocabulary;
my $mode = "${cell_mode}Cell";
my $stack = mx->rnn->SequentialRNNCell();
for my $i (0..$num_layers-1)
{
    my $cell = mx->rnn->$mode(num_hidden => $num_hidden, prefix => "lstm_${i}l0_");
    if($bidirectional)
    {
        $cell = mx->rnn->BidirectionalCell(
            $cell,
            mx->rnn->$mode(
                num_hidden => $num_hidden,
                prefix => "lstm_${i}r0_"
            ),
            output_prefix => "bi_lstm_$i"
        );
    }
    $stack->add($cell);
}

my $data  = mx->sym->Variable('data');
my $label = mx->sym->Variable('softmax_label');
my $embed = mx->sym->Embedding(
        data => $data, input_dim => scalar(keys %vocabulary),
        output_dim => $num_embed, name => 'embed'
);
$stack->reset;
my ($outputs, $states) = $stack->unroll($seq_size, inputs => $embed, merge_outputs => 1);
my $pred  = mx->sym->Reshape($outputs, shape => [-1, $num_hidden*(1+($bidirectional ? 1 : 0))]);
$pred     = mx->sym->FullyConnected(data => $pred, num_hidden => $data_iter->vocab_size, name => 'pred');
$label    = mx->sym->Reshape($label, shape => [-1]);
my $net   = mx->sym->SoftmaxOutput(data => $pred, label => $label, name => 'softmax');

my $contexts;
if(defined $gpus)
{
    $contexts = [map { mx->gpu($_) } split(/,/, $gpus)];
}
else
{
    $contexts = mx->cpu(0);
}

my $model = mx->mod->Module(
    symbol  => $net,
    context => $contexts
);
$model->fit(
    $data_iter,
    eval_metric         => mx->metric->Perplexity,
    kvstore             => $kv_store,
    optimizer           => $optimizer,
    optimizer_params    => {
                                learning_rate => $lr,
                                momentum      => $mom,
                                wd            => $wd,
                                clip_gradient => 5,
                                rescale_grad  => 1/$batch_size,
                                lr_scheduler  => AI::MXNet::FactorScheduler->new(step => 1000, factor => 0.99)
                        },
    initializer         => mx->init->Xavier(factor_type => "in", magnitude => 2.34),
    num_epoch           => $num_epoch,
    batch_end_callback  => mx->callback->Speedometer($batch_size, $disp_batches),
    ($chkp_epoch ? (epoch_end_callback  => [mx->rnn->do_rnn_checkpoint($stack, $chkp_prefix, $chkp_epoch), \&sample]) : ())
);

sub sample {
    return if not $sample_size;
    $model->reshape(data_shapes=>[['data',[1, $seq_size]]], label_shapes=>[['softmax_label',[1, $seq_size]]]);
    my $input = mx->nd->array($fdata->slice([0, $seq_size-1]))->reshape([1, $seq_size]);
    $| = 1;
    for (0..$sample_size-1)
    {
        $model->forward(mx->io->DataBatch(data=>[$input]), is_train => 0);
        my $prob = $model->get_outputs(0)->[0][0]->at($seq_size-1)->aspdl;
        my $next_char = Math::Random::Discrete->new($prob->reshape(-1)->unpdl, [0..scalar(keys %vocabulary)-1])->rand;
        print "$reverse_vocab{$next_char}";
        $input->at(0)->slice([0, $seq_size-2]) .= $input->at(0)->slice([1, $seq_size-1])->copy;
        $input->at(0)->at($seq_size-1) .= $next_char;
    }
    $model->reshape(data_shapes=>[['data',[$batch_size, $seq_size]]], label_shapes=>[['softmax_label',[$batch_size, $seq_size]]]);
}
