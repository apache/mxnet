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
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::AutoGrad qw(autograd);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Base;
use Getopt::Long qw(HelpMessage);

GetOptions(
    'lr=f'           => \(my $lr           = 0.1),
    'log-interval=i' => \(my $log_interval = 100),
    'momentum=f'     => \(my $momentum     = 0.9),
    'hybridize=i'    => \(my $hybridize    = 0  ),
    'cuda=i'         => \(my $cuda         = 0  ),
    'load_params=i'  => \(my $load_params  = 0  ),
    'batch-size=i'   => \(my $batch_size   = 100),
    'epochs=i'       => \(my $epochs       = 1 ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);


# define network

my $net = nn->Sequential();
$net->name_scope(sub {
    $net->add(nn->Dense(128, activation=>'relu'));
    $net->add(nn->Dense(64, activation=>'relu'));
    $net->add(nn->Dense(10));
});
$net->hybridize() if $hybridize;
$net->load_params('mnist.params') if $load_params;
# data

sub transformer
{
    my ($data, $label) = @_;
    $data = $data->reshape([-1])->astype('float32')/255;
    return ($data, $label);
}

my $train_data = gluon->data->DataLoader(
    gluon->data->vision->MNIST('./data', train=>1, transform => \&transformer),
    batch_size=>$batch_size, shuffle=>1, last_batch=>'discard'
);

my $val_data = gluon->data->DataLoader(
    gluon->data->vision->MNIST('./data', train=>0, transform=> \&transformer),
    batch_size=>$batch_size, shuffle=>0
);

# train

sub test
{
    my $ctx = shift;
    my $metric = mx->metric->Accuracy();
    while(defined(my $d = <$val_data>))
    {
        my ($data, $label) = @$d;
        $data = $data->as_in_context($ctx);
        $label = $label->as_in_context($ctx);
        my $output = $net->($data);
        $metric->update([$label], [$output]);
    }
    return $metric->get;
}

sub train
{
    my ($epochs, $ctx) = @_;
    # Collect all parameters from net and its children, then initialize them.
    $net->initialize(mx->init->Xavier(magnitude=>2.24), ctx=>$ctx);
    # Trainer is for updating parameters with gradient.
    my $trainer = gluon->Trainer($net->collect_params(), 'sgd', { learning_rate => $lr, momentum => $momentum });
    my $metric = mx->metric->Accuracy();
    my $loss = gluon->loss->SoftmaxCrossEntropyLoss();

    for my $epoch (0..$epochs-1)
    {
        # reset data iterator and metric at begining of epoch.
        $metric->reset();
        enumerate(sub {
            my ($i, $d) = @_;
            my ($data, $label) = @$d;
            $data = $data->as_in_context($ctx);
            $label = $label->as_in_context($ctx);
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            my $output;
            autograd->record(sub {
                $output = $net->($data);
                my $L = $loss->($output, $label);
                $L->backward;
            });
            # take a gradient step with batch_size equal to data.shape[0]
            $trainer->step($data->shape->[0]);
            # update metric at last.
            $metric->update([$label], [$output]);

            if($i % $log_interval == 0 and $i > 0)
            {
                my ($name, $acc) = $metric->get();
                print "[Epoch $epoch Batch $i] Training: $name=$acc\n";
            }
        }, \@{ $train_data });

        my ($name, $acc) = $metric->get();
        print "[Epoch $epoch] Training: $name=$acc\n";

        my ($val_name, $val_acc) = test($ctx);
        print "[Epoch $epoch] Validation: $val_name=$val_acc\n"
    }
    $net->save_params('mnist.params');
}

train($epochs, $cuda ? mx->gpu(0) : mx->cpu);
