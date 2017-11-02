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
package AI::MXNet::Gluon::NN::Sequential;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Gluon::NN::Sequential
=cut

=head2 DESCRIPTION

    Stacks `Block`s sequentially.

    Example::

        my $net = nn->Sequential()
        # use net's name_scope to give child Blocks appropriate names.
        net->name_scope(sub {
            $net->add($nn->Dense(10, activation=>'relu'));
            $net->add($nn->Dense(20));
        });
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Block';

=head

    Adds block on top of the stack.
=cut

method add(AI::MXNet::Gluon::Block $block)
{
    $self->register_child($block);
}


method forward($x)
{
    for my $block (@{ $self->_children })
    {
        $x = $block->($x);
    }
    return $x;
}

use overload
    '""' => sub
    {
        my $self = shift;
        my $s = "%s(\n{%s}\n)";
        my @blocks;
        my $k = 0;
        for my $v (@{ $self->{_children} })
        {
            push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            $k++;
        }
        sprintf("%s(\n{%s}\n)", $self->_class_name, join("\n", @blocks));
    },
    '@{}' => sub { shift->_children };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::HybridSequential;

=head1 NAME

    AI::MXNet::Gluon::NN::HybridSequential
=cut

=head2 DESCRIPTION

    Stacks `Block`s sequentially.

    Example::

        my $net = nn->Sequential()
        # use net's name_scope to give child Blocks appropriate names.
        net->name_scope(sub {
            $net->add($nn->Dense(10, activation=>'relu'));
            $net->add($nn->Dense(20));
        });
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head

    Adds block on top of the stack.
=cut

method add(AI::MXNet::Gluon::HybridBlock $block)
{
    $self->register_child($block);
}


method forward($x)
{
    for my $block (@{ $self->_children })
    {
        $x = $block->($x);
    }
    return $x;
}

use overload
    '""' => sub
    {
        my $self = shift;
        my $s = "%s(\n{%s}\n)";
        my @blocks;
        my $k = 0;
        for my $v (@{ $self->{_children} })
        {
            push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            $k++;
        }
        sprintf("%s(\n{%s}\n)", $self->_class_name, join("\n", @blocks));
    },
    '@{}' => sub { shift->_children };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Dense;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

method python_constructor_arguments()
{
    ['units'];
}

=head1 NAME

    AI::MXNet::Gluon::NN::Dense

=head1 DESCRIPTION

    Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, weight) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `weight` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use `flatten` to convert it
    to rank 2 manually if necessary.

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    activation : str
        Activation function to use. See help on `Activation` layer.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    flatten : bool, default true
        Whether the input tensor should be flattened.
        If true, all but the first axis of input data are collapsed together.
        If false, all but the last axis of input data are kept the same, and the transformation
        applies on the last axis.
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.

    If flatten is set to be True, then the shapes are:
    Input shape:
        An N-D input with shape
        `(batch_size, x1, x2, ..., xn) with x1 * x2 * ... * xn equal to in_units`.

    Output shape:
        The output would have shape `(batch_size, units)`.

    If ``flatten`` is set to be false, then the shapes are:
    Input shape:
        An N-D input with shape
        `(x1, x2, ..., xn, in_units)`.

    Output shape:
        The output would have shape `(x1, x2, ..., xn, units)`.
=cut

has 'units'               => (is => 'rw', isa => 'Int', required => 1);
has 'activation'          => (is => 'rw', isa => 'Str');
has 'use_bias'            => (is => 'rw', isa => 'Bool', default => 1);
has 'flatten'             => (is => 'rw', isa => 'Bool', default => 1);
has 'weight_initializer'  => (is => 'rw', isa => 'Initializer');
has 'bias_initializer'    => (is => 'rw', isa => 'Initializer', default => 'zeros');
has 'in_units'            => (is => 'rw', isa => 'Int', default => 0);
has [qw/weight bias act/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->weight(
            $self->params->get(
                'weight', shape => [$self->units, $self->in_units],
                init => $self->weight_initializer,
                allow_deferred_init => 1
            )
        );
        if($self->use_bias)
        {
            $self->bias(
                $self->params->get(
                    'bias', shape => [$self->units],
                    init => $self->bias_initializer,
                    allow_deferred_init => 1
                )
            );
        }
        if(defined $self->activation)
        {
            $self->act(
                AI::MXNet::Gluon::NN::Activation->new(
                    activation => $self->activation,
                    prefix => $self->activation.'_'
                )
            );
        }
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$weight, Maybe[GluonInput] :$bias=)
{
    my $act;
    if(not defined $bias)
    {
        $act = $F->FullyConnected($x, $weight, no_bias => 1, num_hidden => $self->units, name => 'fwd');
    }
    else
    {
        $act = $F->FullyConnected($x, $weight, $bias, num_hidden => $self->units, flatten => $self->flatten, name => 'fwd')
    }
    if(defined $self->act)
    {
        $act = $self->act->($act);
    }
    return $act;
}

use overload '""' => sub {
    my $self = shift;
    "${\ $self->_class_name }(${\ $self->units } -> ${\ $self->in_units },"
    ." @{[ $self->act ? $self->act : 'linear' ]})"
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Activation;

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

package AI::MXNet::Gluon::NN::Dropout;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::Dropout
=cut

=head1 DESCRIPTION

    Applies Dropout to the input.

    Dropout consists in randomly setting a fraction `rate` of input units
    to 0 at each update during training time, which helps prevent overfitting.

    Parameters
    ----------
    rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.


    Input shape:
        Arbitrary.

    Output shape:
        Same shape as input.

    References
    ----------
        `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_
=cut
has 'rate' => (is => 'ro', isa => 'Dropout', required => 1);
method python_constructor_arguments() { ['rate'] }

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->Dropout($x, p => $self->rate, name => 'fwd');
}

use overload '""' => sub { my $self = shift; "${\ $self->_class_name }(p = ${\ $self->rate })"; };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::BatchNorm;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::BatchNorm
=cut

=head1 DESCRIPTION

    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Input shape:
        Arbitrary.

    Output shape:
        Same shape as input.
=cut

has 'axis'             => (is => 'ro', isa => 'DimSize',     default => 1);
has 'momentum'         => (is => 'ro', isa => 'Num',         default => 0.9);
has 'epsilon'          => (is => 'ro', isa => 'Num',         default => 1e-5);
has 'center'           => (is => 'ro', isa => 'Bool',        default => 1);
has 'scale'            => (is => 'ro', isa => 'Bool',        default => 1);
has 'beta_initializer' => (is => 'ro', isa => 'Initializer', default => 'zeros');
has [qw/gamma_initializer
        running_mean_initializer
        running_variance_initializer
    /]                 => (is => 'ro', isa => 'Initializer', default => 'ones');
has 'in_channels'      => (is => 'ro', isa => 'DimSize',     default => 0);
has [qw/_kwargs
        gamma
        beta
        running_mean
        running_var/]  => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->_kwargs({
        axis => $self->axis,
        eps => $self->epsilon,
        momentum => $self->momentum,
        fix_gamma => $self->scale ? 0 : 1
    });

    $self->gamma(
        $self->params->get(
            'gamma', grad_req => $self->scale ? 'write' : 'null',
            shape => [$self->in_channels], init => $self->gamma_initializer,
            allow_deferred_init => 1, differentiable => $self->scale
        )
    );
    $self->beta(
        $self->params->get(
            'beta', grad_req => $self->center ? 'write' : 'null',
            shape => [$self->in_channels], init => $self->beta_initializer,
            allow_deferred_init => 1, differentiable => $self->center
        )
    );
    $self->running_mean(
        $self->params->get(
            'running_mean', grad_req => 'null',
            shape => [$self->in_channels], init => $self->running_mean_initializer,
            allow_deferred_init => 1, differentiable => 0
        )
    );
    $self->running_var(
        $self->params->get(
            'running_var', grad_req => $self->center ? 'write' : 'null',
            shape => [$self->in_channels], init => $self->running_variance_initializer,
            allow_deferred_init => 1, differentiable => 0
        )
    );
}

method hybrid_forward(
    GluonClass $F, GluonInput $x,
    GluonInput :$gamma, GluonInput :$beta,
    GluonInput :$running_mean, GluonInput :$running_var
)
{
    return $F->BatchNorm(
        $x, $gamma, $beta, $running_mean, $running_var,
        name =>'fwd', %{ $self->_kwargs }
    );
}

use overload '""' => sub {
    my $self = shift;
    my $f = "%s(%s".($self->in_channels ? ", in_channels=".$self->in_channels : '').')';
    my $content = join(", ", map { join('=', $_, $self->_kwargs->{$_}) } keys %{ $self->_kwargs });
    return sprintf($f, $self->_class_name, $content);
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::LeakyReLU;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::LeakyReLU
=cut

=head1 DESCRIPTION

    Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active

        `f(x) = alpha * x for x < 0`,
        `f(x) = x for x >= 0`.

    Parameters
    ----------
    alpha : float
        slope coefficient for the negative half axis. Must be >= 0.


    Input shape:
        Arbitrary.

    Output shape:
        Same shape as input.
=cut
has 'alpha' => (is => 'ro', isa => 'Num', required => 1);
method python_constructor_arguments()
{
    ['alpha'];
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->LeakyReLU($x, act_type => 'leaky', slope => $self->alpha, name => 'fwd');
}

use overload '""' => sub { my $self = shift; "${\ $self->_class_name }(${\ $self->alpha })"; };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Embedding;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::Embedding
=cut

=head1 DESCRIPTION

    Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]


    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : Initializer
        Initializer for the `embeddings` matrix.


    Input shape:
        2D tensor with shape: `(N, M)`.

    Output shape:
        3D tensor with shape: `(N, M, output_dim)`.
=cut

has [qw/input_dim
    output_dim/]         => (is => 'ro', isa => 'DimSize', required => 1);
has 'dtype'              => (is => 'ro', isa => 'Dtype', default => 'float32');
has 'weight_initalizer'  => (is => 'ro', isa => 'Maybe[Initializer]');
has [qw/_kwargs weight/] => (is => 'rw', init_arg => undef);
method python_constructor_arguments()
{
    ['input_dim', 'output_dim'];
}

sub BUILD
{
    my $self = shift;
    $self->_kwargs({
        input_dim => $self->input_dim,
        output_dim =>  $self->output_dim,
        dtype => $self->dtype
    });
    $self->weight(
        $self->params->get(
            'weight',
            shape => [$self->input_dim, $self->output_dim],
            init => $self->weight_initializer,
            allow_deferred_init => 1
        )
    );
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$weight)
{
    return $F->Embedding($x, $weight, name => 'fwd', %{ $self->_kwargs });
}

use overload '""' => sub {
    my $self = shift;
    "${\ $self->_class_name }(${\ $self->input_dim } -> ${\ $self->output_dim }, ${\ $self->dtype })";
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Flatten;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::Flatten
=cut

=head1 DESCRIPTION

    Flattens the input to two dimensional.

    Input shape:
        Arbitrary shape `(N, a, b, c, ...)`

    Output shape:
        2D tensor with shape: `(N, a*b*c...)`
=cut

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $x->reshape([0, -1]);
}

use overload '""' => sub { shift->_class_name };

__PACKAGE__->register('AI::MXNet::Gluon::NN');

1;