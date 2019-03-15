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

package AI::MXNet::Gluon::NN::Activation;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;

=head1

    AI::MXNet::Gluon::NN::Activation
=cut

=head1 DESCRIPTION

    Applies an activation function to input.

    Parameters
    ----------
    activation : str
        Name of activation function to use.
        See mxnet.ndarray.Activation for available choices.

    Input shape:
        Arbitrary.

    Output shape:
        Same shape as input.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'activation' => (is => 'ro', isa => 'Str', required => 1);

method python_constructor_arguments()
{
    ['activation'];
}

method _alias()
{
    return $self->activation;
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->Activation($x, act_type => $self->activation, name=>'fwd');
}

use overload '""' => sub { my $self = shift; "${\ $self->_class_name }(${\ $self->activation })"; };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::LeakyReLU;
=head1

    AI::MXNet::Gluon::NN::LeakyReLU - Leaky version of a Rectified Linear Unit.
=cut

=head1 DESCRIPTION

    Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active

    Parameters
    ----------
    alpha : float
        slope coefficient for the negative half axis. Must be >= 0.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'alpha' => (is => 'ro', isa => 'Num', required => 1);

method python_constructor_arguments()
{
    ['alpha'];
}

sub BUILD
{
    confess('Slope coefficient for LeakyReLU must be no less than 0')
        unless shift->alpha > 0;
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->LeakyReLU($x, act_type => 'leaky', slope => $self->alpha, name=>'fwd');
}

use overload '""' => sub { my $self = shift; "${\ $self->_class_name }(${\ $self->alpha })"; };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::PReLU;
=head1

    AI::MXNet::Gluon::NN::PReLU - Parametric leaky version of a Rectified Linear Unit.
=cut

=head1 DESCRIPTION

    Parametric leaky version of a Rectified Linear Unit.
    https://arxiv.org/abs/1502.01852

    It learns a gradient when the unit is not active

    Parameters
    ----------
    alpha_initializer : Initializer
        Initializer for the embeddings matrix.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'alpha_initializer' => (is => 'ro', isa => 'Initializer', default => sub { AI::MXNet::Constant->new(0.25) });

method python_constructor_arguments()
{
    ['alpha_initializer'];
}

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->alpha($self->params->get('alpha', shape=>[1], init=>$self->alpha_initializer));
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$alpha)
{
    return $F->LeakyReLU($x, gamma => $alpha, act_type => 'prelu',  name=>'fwd');
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::ELU;
=head1

    AI::MXNet::Gluon::NN::ELU - Exponential Linear Unit (ELU)
=cut

=head1 DESCRIPTION

    Exponential Linear Unit (ELU)
        "Fast and Accurate Deep Network Learning by Exponential Linear Units", Clevert et al, 2016
        https://arxiv.org/abs/1511.07289
        Published as a conference paper at ICLR 2016

    Parameters
    ----------
    alpha : float
        The alpha parameter as described by Clevert et al, 2016
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'alpha' => (is => 'ro', isa => 'Num', default => 1);

method python_constructor_arguments()
{
    ['alpha'];
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->where($x > 0, $x, $self->alpha * ($F->exp($x) - 1));
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::SELU;
=head1

    AI::MXNet::Gluon::NN::SELU - Scaled Exponential Linear Unit (SELU)
=cut

=head1 DESCRIPTION

    Scaled Exponential Linear Unit (SELU)
    "Self-Normalizing Neural Networks", Klambauer et al, 2017
    https://arxiv.org/abs/1706.02515
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $F->LeakyReLU($x, act_type=>'selu', name=>'fwd');
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Swish;
=head1

    AI::MXNet::Gluon::NN::Swish - Swish Activation function
=cut

=head1 DESCRIPTION

    Swish Activation function
        https://arxiv.org/pdf/1710.05941.pdf

    Parameters
    ----------
    beta : float
        swish(x) = x * sigmoid(beta*x)
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'beta' => (is => 'ro', isa => 'Num', default => 1);

method python_constructor_arguments()
{
    ['beta'];
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return return $x * $F->sigmoid($self->beta * $x, name=>'fwd');
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');
