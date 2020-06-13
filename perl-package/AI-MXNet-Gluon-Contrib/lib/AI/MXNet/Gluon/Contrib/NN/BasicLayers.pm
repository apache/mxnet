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
package AI::MXNet::Gluon::Contrib::NN::BasicLayers;

=head1 NAME 

    AI::MXNet::Gluon::Contrib::NN::BasicLayers - An additional collection of Gluon's building blocks.
=cut

use AI::MXNet::Function::Parameters;
package AI::MXNet::Gluon::NN::Concurrent;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Sequential';

=head1 NAME

    AI::MXNet::Gluon::NN::Concurrent - Lays Blocks concurrently.
=cut

=head1 DESCRIPTION

    Lays Blocks concurrently.

    This block feeds its input to all children blocks, and
    produces the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example:

        $net = nn->Concurrent();
        # use net's name_scope to give children blocks appropriate names.
        $net->name_scope(sub {
            $net->add(nn->Dense(10, activation=>'relu'));
            $net->add(nn->Dense(20));
            $net->add(nn->Identity());
        });

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
=cut
has 'axis' => (is => 'rw', isa => 'Int', default => -1);
method python_constructor_arguments() { ['axis'] }

method forward(GluonInput $x)
{
    return AI::MXNet::NDArray->concat((map { $_->($x) } $self->_children->values), dim=>$self->axis);
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::HybridConcurrent;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::HybridSequential';

=head1 NAME

    AI::MXNet::Gluon::NN::HybridConcurrent - Lays HubridBlocks concurrently.
=cut

=head1 DESCRIPTION

    Lays HybridBlocks concurrently.

    This block feeds its input to all children blocks, and
    produces the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example:

        $net = nn->HybridConcurrent();
        # use net's name_scope to give children blocks appropriate names.
        $net->name_scope(sub {
            $net->add(nn->Dense(10, activation=>'relu'));
            $net->add(nn->Dense(20));
            $net->add(nn->Identity());
        });

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
=cut
has 'axis' => (is => 'rw', isa => 'Int', default => -1);
method python_constructor_arguments() { ['axis'] }

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->concat((map { $_->($x) } $self->_children->values), dim=>$self->axis);
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Identity;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::Identity - Block that passes through the input directly.
=cut

=head1 DESCRIPTION

    Block that passes through the input directly.

    This block can be used in conjunction with HybridConcurrent
    block for residual connection.

    Example:

        $net = nn->HybridConcurrent();
        # use net's name_scope to give child Blocks appropriate names.
        $net->name_scope(sub {
            $net->add(nn->Dense(10, activation=>'relu'));
            $net->add(nn->Dense(20));
            $net->add(nn->Identity());
        });
=cut

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $x;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

1;
