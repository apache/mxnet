#!/usr/bin/env perl

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
use Getopt::Long qw(HelpMessage);

GetOptions(
    'print-every=i'  => \(my $print_every  = 100),
    'factor-size=i'  => \(my $factor_size  = 128),
    'use-gpu=i'      => \(my $use_gpu      = 0  ),
    'num-epoch=i'    => \(my $num_epoch    = 3  ),
    'batch-size=i'   => \(my $batch_size   = 128),
    'use-dense=i'    => \(my $use_dense    = 0  ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    train.pl - Run matrix factorization with sparse embedding

=head1 SYNOPSIS

    --print-every  logging frequency, 100
    --factor-size  the factor size of the embedding operation, 128
    --use-gpu      use gpu, 0
    --num-epoch    number of epochs to train, 3
    --batch-size   number of examples per batch, 128
    --use-dense    use the dense embedding operator, 0

=cut

my %MOVIELENS = (
    dataset   => 'ml-10m',
    train     => './data/ml-10M100K/r1.train',
    val       =>  './data/ml-10M100K/r1.test',
    max_user  => 71569,
    max_movie => 65135,
);

sub get_movielens_iter
{
    my ($filename, $batch_size) = @_;
    print "Preparing data iterators for $filename ... \n";
    my @user;
    my @item;
    my @score;
    open(F, $filename) or die $!;
    my $num_samples = 0;
    while(my $line = <F>)
    {
        my @tks = split('::', $line);
        next unless @tks == 4;
        $num_samples++;
        push @user,  [$tks[0]];
        push @item,  [$tks[1]];
        push @score, [$tks[2]];
    }
    # convert to ndarrays
    my $user = mx->nd->array(\@user, dtype=>'int32');
    my $item = mx->nd->array(\@item);
    my $score = mx->nd->array(\@score);
    return mx->io->NDArrayIter(
        data  => Hash::Ordered->new(user => $user, item => $item),
        label => Hash::Ordered->new(score => $score),
        batch_size => $batch_size,
        shuffle => 1
    );
}

sub matrix_fact_net
{
    my ($factor_size, $num_hidden, $max_user, $max_item, $sparse_embed) = @_;
    $sparse_embed //= 1;
    # input
    my $user = mx->symbol->Variable('user');
    my $item = mx->symbol->Variable('item');
    my $score = mx->symbol->Variable('score');
    if($sparse_embed)
    {
        # user feature lookup
        my $user_weight = mx->symbol->Variable('user_weight', stype=>'row_sparse');
        $user = mx->symbol->contrib->SparseEmbedding(data=>$user, weight=>$user_weight,
                                                 input_dim=>$max_user, output_dim=>$factor_size);
        # item feature lookup
        my $item_weight = mx->symbol->Variable('item_weight', stype=>'row_sparse');
        $item = mx->symbol->contrib->SparseEmbedding(data=>$item, weight=>$item_weight,
                                                 input_dim=>$max_item, output_dim=>$factor_size);
    }
    else
    {
        # user feature lookup
        $user = mx->symbol->Embedding(data=>$user, input_dim=>$max_user, output_dim=>$factor_size);
        # item feature lookup
        $item = mx->symbol->Embedding(data=>$item, input_dim=>$max_item, output_dim=>$factor_size);
    }
    # non-linear transformation of user features
    $user = mx->symbol->Activation(data=>$user, act_type=>'relu');
    $user = mx->symbol->FullyConnected(data=>$user, num_hidden=>$num_hidden);
    # non-linear transformation of item features
    $item = mx->symbol->Activation(data=>$item, act_type=>'relu');
    $item = mx->symbol->FullyConnected(data=>$item, num_hidden=>$num_hidden);
    # predict by the inner product, which is elementwise product and then sum
    my $pred = $user * $item;
    $pred = mx->symbol->sum(data=>$pred, axis => 1);
    $pred = mx->symbol->Flatten(data=>$pred);
    # loss layer
    $pred = mx->symbol->LinearRegressionOutput(data=>$pred, label=>$score);
    return $pred;
}

my $optimizer = 'sgd';
my $use_sparse = not $use_dense;

my $momentum = 0.9;
my $ctx = $use_gpu ? mx->gpu(0) : mx->cpu(0);
my $learning_rate = 0.1;

# prepare dataset and iterators
my $max_user   = $MOVIELENS{max_user};
my $max_movies = $MOVIELENS{max_movie};
my $train_iter = get_movielens_iter($MOVIELENS{train}, $batch_size);
my $val_iter   = get_movielens_iter($MOVIELENS{val}  , $batch_size);

# construct the model
my $net = matrix_fact_net($factor_size, $factor_size, $max_user, $max_movies, $use_sparse);

# initialize the module
my $mod = mx->module->Module(symbol=>$net, context=>$ctx, data_names=>['user', 'item'],
                           label_names=>['score']);
$mod->bind(data_shapes=>$train_iter->provide_data, label_shapes=>$train_iter->provide_label);
$mod->init_params(initializer=>mx->init->Xavier(factor_type=>"in", magnitude=>2.34));
my $optim = mx->optimizer->create($optimizer, learning_rate=>$learning_rate, momentum=>$momentum,
                                wd=>1e-4, rescale_grad=>1.0/$batch_size);
$mod->init_optimizer(optimizer=>$optim);

# use MSE as the metric
my $metric = mx->metric->create(['MSE']);
my $speedometer = mx->callback->Speedometer($batch_size, $print_every);
print "Training started ...\n";
for my $epoch (0..$num_epoch-1)
{
    my $nbatch = 0;
    $metric->reset();
    while(my $batch = <$train_iter>)
    {
        $nbatch += 1;
        $mod->forward_backward($batch);
        # update all parameters
        $mod->update();
        # update training metric
        $mod->update_metric($metric, $batch->label);
        my $speedometer_param = AI::MXNet::BatchEndParam->new(
            epoch=>$epoch, nbatch=>$nbatch,
            eval_metric=>$metric
        );
        $speedometer->($speedometer_param);
    }
    # evaluate metric on validation dataset
    my $score = $mod->score($val_iter, ['MSE']);
    printf("epoch %d, eval MSE = %s \n", $epoch, $score->{mse});
    # reset the iterator for next pass of data
    $train_iter->reset();
    $val_iter->reset();
}
print "Training completed.\n";

