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

=head2

    Adds block on top of the stack.
=cut

method add(AI::MXNet::Gluon::Block @block)
{
    $self->register_child($_) for @block;
}


method forward($x)
{
    for my $block ($self->_children->values)
    {
        $x = $block->($x);
    }
    return $x;
}

use overload
    '""' => sub
    {
        my $self = shift;
        my $s = "%s(\n%s\n)";
        my @blocks;
        my $k = 0;
        for my $v ($self->_children->values)
        {
            push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            $k++;
        }
        sprintf("%s(\n%s\n)", $self->_class_name, join("\n", @blocks));
    },
    '@{}' => sub { [shift->_children->values] };

method slice(Slice $slice)
{
    my $new = __PACKAGE__->new;
    $new->add(@{ $self }[ @$slice ]);
    return $new;
}

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

=head2

    Adds block on top of the stack.
=cut

method add(AI::MXNet::Gluon::HybridBlock @block)
{
    $self->register_child($_) for @block;
}


method hybrid_forward($F, $x)
{
    for my $block ($self->_children->values)
    {
        $x = $block->($x);
    }
    return $x;
}

use overload
    '""' => sub
    {
        my $self = shift;
        my $s = "%s(\n%s\n)";
        my @blocks;
        my $k = 0;
        for my $v ($self->_children->values)
        {
            push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            $k++;
        }
        sprintf("%s(\n%s\n)", $self->_class_name, join("\n", @blocks));
    },
    '@{}' => sub { [shift->_children->values] };

method slice(Slice $slice)
{
    my $new = __PACKAGE__->new;
    $new->add(@{ $self }[ @$slice ]);
    return $new;
}

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
                AI::MXNet::Gluon::NN->Activation(
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
    sparse_grad: bool
        If True, gradient w.r.t. weight will be a 'row_sparse' NDArray.
=cut

has [qw/input_dim
    output_dim/]         => (is => 'ro', isa => 'DimSize', required => 1);
has 'dtype'              => (is => 'ro', isa => 'Dtype', default => 'float32');
has 'weight_initalizer'  => (is => 'ro', isa => 'Maybe[Initializer]');
has 'sparse_grad'        => (is => 'ro', isa => 'Bool', default => 0);
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
        dtype => $self->dtype,
        sparse_grad => $self->sparse_grad
    });
    $self->weight(
        $self->params->get(
            'weight',
            shape => [$self->input_dim, $self->output_dim],
            init => $self->weight_initializer,
            allow_deferred_init => 1,
            dtype => $self->dtype,
            grad_stype => ($self->sparse_grad ? 'row_sparse' : 'default')
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

package AI::MXNet::Gluon::NN::InstanceNorm;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::InstanceNorm - Applies instance normalization to the n-dimensional input array.
=cut

=head1 DESCRIPTION

    Applies instance normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array where (n>2) and normalizes
    the input using the following formula:

    Parameters
    ----------
    axis : int, default 1
        The axis that will be excluded in the normalization process. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `InstanceNorm`. If `layout='NHWC'`, then set `axis=3`. Data will be
        normalized along axes excluding the first axis and the axis given.
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
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.

    References
    ----------
        Instance Normalization: The Missing Ingredient for Fast Stylization
        <https://arxiv.org/abs/1607.08022>

    Examples
    --------
    >>> # Input of shape (2,1,2)
    >>> $x = mx->nd->array([[[ 1.1,  2.2]],
    ...                 [[ 3.3,  4.4]]]);
    >>> $layer = nn->InstanceNorm()
    >>> $layer->initialize(ctx=>mx->cpu(0))
    >>> $layer->($x)
    [[[-0.99998355  0.99998331]]
     [[-0.99998319  0.99998361]]]
    <NDArray 2x1x2 @cpu(0)>
=cut

has 'axis'              => (is => 'ro', isa => 'Int',  default => 1);
has 'epsilon'           => (is => 'ro', isa => 'Num',  default => 1e-5);
has 'center'            => (is => 'ro', isa => 'Bool', default => 1);
has 'scale'             => (is => 'ro', isa => 'Bool', default => 0);
has 'beta_initializer'  => (is => 'rw', isa => 'Initializer', default => 'zeros');
has 'gamma_initializer' => (is => 'rw', isa => 'Initializer', default => 'ones');
has 'in_channels'       => (is => 'rw', isa => 'Int',  default => 0);
has [qw/_kwargs
        gamma beta/]    => (is => 'rw', init_arg => undef);
method python_constructor_arguments()
{
    [qw/axis epsilon center scale beta_initializer gamma_initializer in_channels/];
}


sub BUILD
{
    my $self = shift;
    $self->_kwargs(Hash::Ordered->new(eps => $self->epsilon, axis => $self->axis, center => $self->center, scale => $self->scale));
    $self->gamma(
        $self->params->get(
            'gamma', grad_req => $self->scale ? 'write' :'null',
            shape => [$self->in_channels], init => $self->gamma_initializer,
            allow_deferred_init => 1
        )
    );
    $self->beta(
        $self->params->get(
            'beta', grad_req => $self->scale ? 'write' :'null',
            shape => [$self->in_channels], init => $self->beta_initializer,
            allow_deferred_init => 1
        )
    );
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$gamma, GluonInput :$beta)
{
    if($self->axis == 1)
    {
        return $F->InstanceNorm(
                    $x, $gamma, $beta,
                    name=>'fwd', eps=>$self->epsilon
        );
    }
    $x = $x->swapaxes(1, $self->axis);
    return $F->InstanceNorm(
                    $x, $gamma, $beta, name=>'fwd',
                    eps => $self->epsilon
    )->swapaxes(1, $self->axis);
}

use overload '""' => sub {
    my $self = shift;
    my $in_channels = ", in_channels=${\ $self->in_channels }";
    my $content = join(', ', map { join('=', $_, $self->_kwargs->get($_)) } $self->_kwargs->keys);
    return "${\ $self->_class_name }($content, $in_channels)";
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::LayerNorm;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::LayerNorm - Applies layer normalization to the n-dimensional input array.
=cut

=head1 DESCRIPTION

    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    Parameters
    ----------
    axis : int, default -1
        The axis that should be normalized. This is typically the axis of the channels.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.

    References
    ----------
        `Layer Normalization
        <https://arxiv.org/pdf/1607.06450.pdf>`_

    Examples
    --------
    >>> # Input of shape (2, 5)
    >>> $x = mx->nd->array([[1, 2, 3, 4, 5], [1, 1, 2, 2, 2]])
    >>> # Layer normalization is calculated with the above formula
    >>> $layer = nn->LayerNorm()
    >>> $layer->initialize(ctx=>mx->cpu(0))
    >>> $layer->($x)
    [[-1.41421    -0.707105    0.          0.707105    1.41421   ]
     [-1.2247195  -1.2247195   0.81647956  0.81647956  0.81647956]]
    <NDArray 2x5 @cpu(0)>
=cut

has 'axis'              => (is => 'ro', isa => 'Int',  default => -1);
has 'epsilon'          => (is => 'ro', isa => 'Num',  default => 1e-5);
has 'center'            => (is => 'ro', isa => 'Bool', default => 1);
has 'scale'             => (is => 'ro', isa => 'Bool', default => 0);
has 'beta_initializer'  => (is => 'rw', isa => 'Initializer', default => 'zeros');
has 'gamma_initializer' => (is => 'rw', isa => 'Initializer', default => 'ones');
has 'in_channels'       => (is => 'rw', isa => 'Int',  default => 0);
has [qw/_kwargs
        gamma beta/]    => (is => 'rw', init_arg => undef);
method python_constructor_arguments()
{
    [qw/axis epsilon center scale beta_initializer gamma_initializer in_channels/];
}

sub BUILD
{
    my $self = shift;
    $self->_kwargs(Hash::Ordered->new(eps => $self->epsilon, axis => $self->axis, center => $self->center, scale => $self->scale));
    $self->gamma(
        $self->params->get(
            'gamma', grad_req => $self->scale ? 'write' :'null',
            shape => [$self->in_channels], init => $self->gamma_initializer,
            allow_deferred_init => 1
        )
    );
    $self->beta(
        $self->params->get(
            'beta', grad_req => $self->scale ? 'write' :'null',
            shape => [$self->in_channels], init => $self->beta_initializer,
            allow_deferred_init => 1
        )
    );
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$gamma, GluonInput :$beta)
{
    return $F->LayerNorm(
        $x, $gamma, $beta,
        eps => $self->epsilon, axis => $self->axis
    );
}

use overload '""' => sub {
    my $self = shift;
    my $in_channels = ", in_channels=${\ $self->in_channels }";
    my $content = join(', ', map { join('=', $_, $self->_kwargs->get($_)) } $self->_kwargs->keys);
    return "${\ $self->_class_name }($content, $in_channels)";
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Lambda;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Block';

=head1 NAME

    AI::MXNet::Gluon::NN::Lambda - Wraps an operator or an expression as a Block object.
=cut

=head1 DESCRIPTION

    Wraps an operator or an expression as a Block object.

    Parameters
    ----------
    function : str or sub
        Function used in lambda must be one of the following:
        1) the name of an operator that is available in ndarray. For example

            $block = nn->Lambda('tanh')

        2) a sub. For example

            $block = nn->Lambda(sub { my $x = shift; nd->LeakyReLU($x, slope=>0.1) });
=cut

has '_func_impl' => (is => 'rw', init_arg => 'function', isa => 'Str|CodeRef', required => 1);
has '_func_name' => (is => 'rw', init_arg => undef, default => 'custom_sub');
method python_constructor_arguments() { ['function'] }

sub BUILD
{
    my $self = shift;
    if(not ref $self->_func_impl)
    {
        confess("Function name ${\ $self->_func_impl } is not found in ndarray.")
            unless AI::MXNet::NDArray->can($self->_func_impl);
        $self->_func_name($self->_func_impl);
        my $f = $self->_func_impl;
        $self->_func_impl(sub { return AI::MXNet::NDArray->$f(@_) });
    }
}

method forward(@args)
{
    return $self->_func_impl->(@args);
}

use overload '""' => sub {
    my $self = shift;
    return "${\ $self->_class_name }(${\ $self->_func_name })";
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::HybridLambda;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::NN::HybridLambda - Wraps an operator or an expression as a HybridBlock object.
=cut

=head1 DESCRIPTION

    Wraps an operator or an expression as a HybridBlock object.

    Parameters
    ----------
    function : str or sub
        Function used in lambda must be one of the following:
        1) the name of an operator that is available in symbol and ndarray. For example

            $block = nn->Lambda('tanh')

        2) a sub. For example

            $block = nn->Lambda(sub { my $F = shift; $F->LeakyReLU($x, slope=>0.1) });
=cut

has '_func_impl' => (is => 'rw', init_arg => 'function', isa => 'Str|CodeRef', required => 1);
has '_func_name' => (is => 'rw', init_arg => undef, default => 'custom_sub');
method python_constructor_arguments() { ['function'] }

sub BUILD
{
    my $self = shift;
    if(not ref $self->_func_impl)
    {
        confess("Function name ${\ $self->_func_impl } is not found in ndarray.")
            unless AI::MXNet::NDArray->can($self->_func_impl) or AI::MXNet::Symbol->can($self->_func_impl);
        $self->_func_name($self->_func_impl);
        my $f = $self->_func_impl;
        $self->_func_impl(sub { my $F = shift; return $F->$f(@_) });
    }
}

method hybrid_forward(@args)
{
    return $self->_func_impl->(@args);
}

use overload '""' => sub {
    my $self = shift;
    return "${\ $self->_class_name }(${\ $self->_func_name })";
};

__PACKAGE__->register('AI::MXNet::Gluon::NN');

1;
