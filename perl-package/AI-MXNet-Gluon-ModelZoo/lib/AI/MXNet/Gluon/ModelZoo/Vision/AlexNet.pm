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

package AI::MXNet::Gluon::ModelZoo::Vision::AlexNet;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::AlexNet - AlexNet model from the `"One weird trick..."
=cut

=head1 DESCRIPTION

    AlexNet model from the "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.

    Parameters
    ----------
    classes : Int, default 1000
        Number of classes for the output layer.
=cut
has 'classes' => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments() { ['classes'] }

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        $self->features->name_scope(sub {
            $self->features->add(nn->Conv2D(64, kernel_size=>11, strides=>4,
                                            padding=>2, activation=>'relu'));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2));
            $self->features->add(nn->Conv2D(192, kernel_size=>5, padding=>2,
                                            activation=>'relu'));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2));
            $self->features->add(nn->Conv2D(384, kernel_size=>3, padding=>1,
                                            activation=>'relu'));
            $self->features->add(nn->Conv2D(256, kernel_size=>3, padding=>1,
                                            activation=>'relu'));
            $self->features->add(nn->Conv2D(256, kernel_size=>3, padding=>1,
                                            activation=>'relu'));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2));
            $self->features->add(nn->Flatten());
            $self->features->add(nn->Dense(4096, activation=>'relu'));
            $self->features->add(nn->Dropout(0.5));
            $self->features->add(nn->Dense(4096, activation=>'relu'));
            $self->features->add(nn->Dropout(0.5));
        });
        $self->output(nn->Dense($self->classes));
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision;

=head2 alexnet

    AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method alexnet(
    Bool :$pretrained=0,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models',
    Int :$classes=1000
)
{
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::AlexNet->new($classes);
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                'alexnet',
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

1;
