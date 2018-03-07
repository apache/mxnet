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
use AI::MXNet::Base qw(pdl enumerate digitize hash array_index range);
use Getopt::Long qw(HelpMessage);

GetOptions(
    'log-interval=i' => \(my $log_interval = 100),
    'optimizer=s'    => \(my $optimizer    = 'adam'),
    'cuda'           => \(my $cuda         = 0  ),
    'num-epoch=i'    => \(my $num_epoch    = 10  ),
    'batch-size=i'   => \(my $batch_size   = 100),
    'lr=f'           => \(my $lr    = 0.001     ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    train.pl - Run sparse wide and deep classification

=head1 SYNOPSIS

    --log-interval number of batches to wait before logging training status, 100
    --optimizer    what optimizer to use, 'adam'
    --cuda         train on gpu with cuda, 0
    --num-epoch    number of epochs to train, 10
    --batch-size   number of examples per batch, 100
    --lr           learning rate, 0.001

=cut

my %allowed_optimizers = qw(adam 1 sgd 1 ftrl 1);
Carp::confess("optimizer can only be one of 'adam', 'sgd', 'ftrl'") 
    unless exists $allowed_optimizers{ $optimizer };

sub wide_deep_model
{
    my ($num_linear_features, $num_embed_features, $num_cont_features,
                    $input_dims, $hidden_units) = @_;
    # wide model
    my $csr_data = mx->symbol->Variable("csr_data", stype=>'csr');
    my $label = mx->symbol->Variable("softmax_label");

    my $norm_init = mx->initializer->Normal(sigma=>0.01);
    # weight with row_sparse storage type to enable sparse gradient updates
    my $weight = mx->symbol->Variable("linear_weight", shape=>[$num_linear_features, 2],
                                init=>$norm_init, stype=>'row_sparse');
    my $bias = mx->symbol->Variable("linear_bias", shape=>[2]);
    my $dot = mx->symbol->sparse->dot($csr_data, $weight);
    my $linear_out = mx->symbol->broadcast_add($dot, $bias);
    # deep model
    my $dns_data = mx->symbol->Variable("dns_data");
    # embedding features
    my $x = mx->symbol->slice(data=>$dns_data, begin=>[0, 0],
                        end=>[undef, $num_embed_features]);
    my $embeds = mx->symbol->split(data=>$x, num_outputs=>$num_embed_features, squeeze_axis=>1);
    # continuous features
    $x = mx->symbol->slice(data=>$dns_data, begin=>[0, $num_embed_features],
                        end=>[undef, $num_embed_features + $num_cont_features]);
    my @features = ($x);

    enumerate(sub {
        my ($i, $embed) = @_;
        my $embed_weight = mx->symbol->Variable("embed_${i}_weight", stype=>'row_sparse');
        push @features, mx->symbol->contrib->SparseEmbedding(data=>$embed, weight=>$embed_weight,
                        input_dim=>$input_dims->[$i], output_dim=>$hidden_units->[0]);

    }, $embeds);

    my $hidden = mx->symbol->concat(@features, dim=>1);
    $hidden = mx->symbol->FullyConnected(data=>$hidden, num_hidden=>$hidden_units->[1]);
    $hidden = mx->symbol->Activation(data=>$hidden, act_type=>'relu');
    $hidden = mx->symbol->FullyConnected(data=>$hidden, num_hidden=>$hidden_units->[2]);
    $hidden = mx->symbol->Activation(data=>$hidden, act_type=>'relu');
    my $deep_out = mx->symbol->FullyConnected(data=>$hidden, num_hidden=>2);

    my $out = mx->symbol->SoftmaxOutput($linear_out + $deep_out, $label, name=>'model');
    return $out;
}

sub preprocess_uci_adult
{
    my ($data_name) = @_;
    # Some tricks of feature engineering are adapted
    # from tensorflow's wide and deep tutorial.
    my @csv_columns = (
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "gender",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket"
    );

    my %vocabulary_dict = (
        "gender" => [
            "Female", "Male"
        ],
        "education" => [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
        ],
        "marital_status" => [
            "Married-civ-spouse", "Divorced", "Married-spouse-absent",
            "Never-married", "Separated", "Married-AF-spouse", "Widowed"
        ],
        "relationship" => [
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
            "Other-relative"
        ],
        "workclass" => [
            "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
            "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
        ]
    );
    # wide columns
    my @crossed_columns = (
        ["education", "occupation"],
        ["native_country", "occupation"],
        ["age_buckets", "education", "occupation"],
    );
    my @age_boundaries = (18, 25, 30, 35, 40, 45, 50, 55, 60, 65);
    # deep columns
    my @indicator_columns = ('workclass', 'education', 'gender', 'relationship');

    my @embedding_columns = ('native_country', 'occupation');

    my @continuous_columns = ('age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week');
    # income_bracket column is the label
    my %labels = ("<=50K" => 0, ">50K" => 1);

    my $hash_bucket_size = 1000;

    my $csr_ncols = @crossed_columns * $hash_bucket_size;
    my $dns_ncols = @continuous_columns + @embedding_columns;
    for my $col (@indicator_columns)
    {
        $dns_ncols += @{ $vocabulary_dict{$col} };
    }

    my @label_list;
    my @csr_list;
    my @dns_list;

    open(F, $data_name) or die $!;
    while(<F>)
    {
        chomp;
        my %row;
        @row{ @csv_columns } = split(/\s*,\s*/);
        next if not defined $row{income_bracket};
        push @label_list, $labels{$row{income_bracket}};
        enumerate(sub {
            my ($i, $cols) = @_;
            if($cols->[0] eq "age_buckets")
            {
                my $age_bucket = digitize($row{age}, \@age_boundaries);
                my $s = join('_', map { $row{$_} } @{ $cols }[1..@{$cols}-1]);
                $s .= '_' . $age_bucket;
                push @csr_list, [$i * $hash_bucket_size + hash($s) % $hash_bucket_size, 1];
            }
            else
            {
                my $s = join('_', map { $row{$_} } @{ $cols });
                push @csr_list, [$i * $hash_bucket_size + hash($s) % $hash_bucket_size, 1];
            }
        }, \@crossed_columns);
        my @dns_row = (0) x $dns_ncols;
        my $dns_dim = 0;
        for my $col (@embedding_columns)
        {
            $dns_row[$dns_dim] = hash($row{$col}) % $hash_bucket_size;
            $dns_dim += 1;
        }

        for my $col (@indicator_columns)
        {
            $dns_row[$dns_dim + array_index($row{$col}, $vocabulary_dict{$col})] = 1;
            $dns_dim += @{$vocabulary_dict{$col}};
        }

        for my $col (@continuous_columns)
        {
            $dns_row[$dns_dim] = $row{col};
            $dns_dim += 1;
        }
        push @dns_list, \@dns_row;
    }
    my @data_list = map { $_->[1] } @csr_list;
    my @indices_list = map { $_->[0] } @csr_list;
    my @indptr_list = range(0, @indices_list + 1, scalar @crossed_columns);
    # convert to ndarrays
    my $csr = mx->nd->sparse->csr_matrix([\@data_list, \@indices_list, \@indptr_list],
                                  shape=>[scalar @label_list, $hash_bucket_size * @crossed_columns]);
    my $dns = pdl(\@dns_list);
    my $label = pdl(\@label_list);
    return ($csr, $dns, $label);
}

# Related to feature engineering, please see preprocess in data.py
my %ADULT = (
    train => './data/adult.data',
    test  => './data/adult.test',
    num_linear_features => 3000,
    num_embed_features => 2,
    num_cont_features => 38,
    embed_input_dims => [1000, 1000],
    hidden_units => [8, 50, 100],
);


my $ctx = $cuda ? mx->gpu : mx->cpu;

# dataset
my ($train_csr, $train_dns, $train_label) = preprocess_uci_adult($ADULT{train});
my ($val_csr, $val_dns, $val_label) = preprocess_uci_adult($ADULT{test});
my $model = wide_deep_model(
    $ADULT{num_linear_features}, $ADULT{num_embed_features},
    $ADULT{num_cont_features}, $ADULT{embed_input_dims},
    $ADULT{hidden_units}
);

# data iterator
my $train_data = mx->io->NDArrayIter(
    data  => Hash::Ordered->new(csr_data => $train_csr, dns_data => $train_dns),
    label => Hash::Ordered->new(softmax_label => $train_label), 
    batch_size => $batch_size,
    shuffle => 1,
    last_batch_handle => 'discard'
);
my $eval_data = mx->io->NDArrayIter(
    data  => Hash::Ordered->new(csr_data => $val_csr, val_data => $val_dns),
    label => Hash::Ordered->new(softmax_label => $val_label), 
    batch_size => $batch_size,
    shuffle => 0,
    last_batch_handle => 'discard'
);

# module
my $mod = mx->mod->Module(
    symbol => $model, context => $ctx, data_names=>['csr_data', 'dns_data'],
    label_names => ['softmax_label']
);
$mod->bind(data_shapes => $train_data->provide_data, label_shapes => $train_data->provide_label);
$mod->init_params();
my $optim = mx->optimizer->create($optimizer, learning_rate=>$lr, rescale_grad=>1/$batch_size);
$mod->init_optimizer(optimizer=>$optim);
# use accuracy as the metric
my $metric = mx->metric->create('acc');
# get the sparse weight parameter
my $speedometer = mx->callback->Speedometer($batch_size, $log_interval);

print "Training started ...\n";

for my $epoch (0..$num_epoch-1)
{
    my $nbatch = 0;
    $metric->reset;
    while(my $batch = <$train_data>)
    {
        $nbatch++;
        $mod->forward_backward($batch);
        # update all parameters (including the weight parameter)
        $mod->update;
        # update training metric
        $mod->update_metric($metric, $batch->label);
        my $speedometer_param = AI::MXNet::BatchEndParam->new(
            epoch=>$epoch, nbatch=>$nbatch,
            eval_metric=>$metric
        );
        $speedometer->($speedometer_param);
    }
    # evaluate metric on validation dataset
    my $score = $mod->score($eval_data, 'acc');
    printf("epoch %d, validation accuracy = %.4f\n", $epoch, $score->{accuracy});

    $mod->save_checkpoint("checkpoint", $epoch, 1);
    # reset the iterator for next pass of data
    $train_data->reset;
}

print "Training completed.\n";
