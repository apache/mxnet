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

package AI::MXNet::Gluon;
use strict;
use warnings;
use AI::MXNet::NS 'global';
use AI::MXNet::Gluon::Loss 'loss';
use AI::MXNet::Gluon::Trainer;
use AI::MXNet::Gluon::Utils;
use AI::MXNet::Gluon::Data 'data';
use AI::MXNet::Gluon::NN 'nn';
use AI::MXNet::Gluon::RNN 'rnn';

sub utils { 'AI::MXNet::Gluon::Utils' }
sub model_zoo { require AI::MXNet::Gluon::ModelZoo; 'AI::MXNet::Gluon::ModelZoo' }

=head1 NAME

    AI::MXNet::Gluon - High-level interface for MXNet.
=cut

=head1 DESCRIPTION

    The AI::MXNet::Gluon package is a high-level interface for MXNet designed to be easy to use,
    while keeping most of the flexibility of a low level API.
    AI::MXNet::Gluon supports both imperative and symbolic programming,
    making it easy to train complex models imperatively in Perl.

    Based on the Gluon API specification,
    the Gluon API in Apache MXNet provides a clear, concise, and simple API for deep learning.
    It makes it easy to prototype, build, and train deep learning models without sacrificing training speed.

    Advantages.

    Simple, Easy-to-Understand Code: Gluon offers a full set of plug-and-play neural network building blocks,
    including predefined layers, optimizers, and initializers.

    Flexible, Imperative Structure: Gluon does not require the neural network model to be rigidly defined,
    but rather brings the training algorithm and model closer together to provide flexibility in the development process.

    Dynamic Graphs: Gluon enables developers to define neural network models that are dynamic,
    meaning they can be built on the fly, with any structure, and using any of Perl's native control flow.

    High Performance: Gluon provides all of the above benefits without impacting the training speed that the underlying engine provides.


    Simple, Easy-to-Understand Code
    Use plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers:

    use AI::MXNet qw(mx);
    use AI::MXNet::Gluon qw(gluon);

    my $net = gluon->nn->Sequential;
    # When instantiated, Sequential stores a chain of neural network layers.
    # Once presented with data, Sequential executes each layer in turn, using
    # the output of one layer as the input for the next
    $net->name_scope(sub {
        $net->add(gluon->nn->Dense(256, activation=>"relu")); # 1st layer (256 nodes)
        $net->add(gluon->nn->Dense(256, activation=>"relu")); # 2nd hidden layer
        $net->add(gluon->nn->Dense($num_outputs));
    });

    Flexible, Imperative Structure.

    Prototype, build, and train neural networks in fully imperative manner using the AI::MXNet::MXNet package and the Gluon trainer method:

    use AI::MXNet::Base; # provides helpers, such as zip, enumerate, etc.
    use AI::MXNet::AutoGrad qw(autograd);
    my $epochs = 10;

    for(1..$epochs)
    {
        for(zip($train_data))
        {
            my ($data, $label) = @$_;
            autograd->record(sub {
                my $output = $net->($data); # the forward iteration
                my $loss = gluon->loss->softmax_cross_entropy($output, $label);
                $loss->backward;
            });
            $trainer->step($data->shape->[0]); ## batch size
        }
    }

    Dynamic Graphs.

    Build neural networks on the fly for use cases where neural networks must change in size and shape during model training:

    use AI::MXNet::Function::Parameters;

    method forward(GluonClass $F, GluonInput $inputs, GluonInput :$tree)
    {
        my $children_outputs = [
            map { $self->forward($F, $inputs, $_) @{ $tree->children }
        ];
        #Recursively builds the neural network based on each input sentence’s
        #syntactic structure during the model definition and training process
        ...
    }

    High Performance

    Easily cache the neural network to achieve high performance by defining your neural network with HybridSequential
    and calling the hybridize method:

    use AI::MXNet::Gluon::NN qw(nn);

    my $net = nn->HybridSequential;
    $net->name_scope(sub {
        $net->add(nn->Dense(256, activation=>"relu"));
        $net->add(nn->Dense(128, activation=>"relu"));
        $net->add(nn->Dense(2));
    });

    $net->hybridize();
    See more at L<Python docs|https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/index.html>
=cut

1;
