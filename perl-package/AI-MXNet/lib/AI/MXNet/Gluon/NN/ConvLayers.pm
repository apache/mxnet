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

package AI::MXNet::Gluon::NN::Conv;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Symbol;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

func _infer_weight_shape($op_name, $data_shape, $kwargs)
{
    my $sym = AI::MXNet::Symbol->$op_name(
        AI::MXNet::Symbol->var('data', shape => $data_shape), %{ $kwargs }
    );
    return ($sym->infer_shape_partial)[0];
}

=head1 NAME

    AI::MXNet::Gluon::NN::Conv
=cut

=head1 DESCRIPTION

    Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is `True`, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: int or tuple/list of n ints,
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
=cut

has 'channels'           => (is => 'rw', isa => 'Int', required => 1);
has 'in_channels'        => (is => 'rw', isa => 'Int', default => 0);
has 'kernel_size'        => (is => 'rw', isa => 'DimSize|Shape', required => 1);
has [qw/strides
        padding
        dilation/]       => (is => 'rw', isa => 'DimSize|Shape');
has 'groups'             => (is => 'rw', isa => 'Int');
has [qw/layout
    activation/]         => (is => 'rw', isa => 'Str');
has 'op_name'            => (is => 'rw', isa => 'Str', default => 'Convolution');
has 'use_bias'           => (is => 'rw', isa => 'Bool', default => 1);
has 'weight_initializer' => (is => 'rw', isa => 'Maybe[Initializer]');
has 'bias_initializer'   => (is => 'rw', isa => 'Maybe[Initializer]', default => 'zeros');
has 'adj'                => (is => 'rw');
has [qw/weight bias
        kwargs act/]     => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { [qw/channels kernel_size strides padding dilation/] }

sub BUILD
{
    my $self = shift;
    $self->_update_kernel_size;
    $self->name_scope(sub {
        if(not ref $self->strides)
        {
            $self->strides([($self->strides) x @{ $self->kernel_size }]);
        }
        if(not ref $self->padding)
        {
            $self->padding([($self->padding) x @{ $self->kernel_size }]);
        }
        if(not ref $self->dilation)
        {
            $self->dilation([($self->dilation) x @{ $self->kernel_size }]);
        }
        $self->kwargs({
            kernel => $self->kernel_size, stride => $self->strides, dilate => $self->dilation,
            pad => $self->padding, num_filter => $self->channels, num_group => $self->groups,
            no_bias => $self->use_bias ? 0 : 1, layout => $self->layout
        });
        if(defined $self->adj)
        {
            $self->kwargs->{adj} = $self->adj;
        }

        my @dshape = (0)x(@{ $self->kernel_size } + 2);
        $dshape[index($self->layout, 'N')] = 1;
        $dshape[index($self->layout, 'C')] = $self->in_channels;
        my $wshapes = _infer_weight_shape($self->op_name, \@dshape, $self->kwargs);
        $self->weight(
            $self->params->get(
                'weight', shape => $wshapes->[1],
                init => $self->weight_initializer,
                allow_deferred_init => 1
            )
        );
        if($self->use_bias)
        {
            $self->bias(
                $self->params->get(
                    'bias', shape => $wshapes->[2],
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
                    prefix     => $self->activation.'_'
                )
            );
        }
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x, GluonInput :$weight, Maybe[GluonInput] :$bias=)
{
    my $op_name = $self->op_name;
    my $act = $F->$op_name($x, $weight, defined $bias ? $bias : (), name => 'fwd', %{ $self->kwargs });
    if(defined $self->act)
    {
        $act = $self->act->($act);
    }
    return $act;
}

method _alias() { 'conv' }

use Data::Dumper;
use overload '""' => sub {
    my $self = shift;
    my $s = '%s(%s, kernel_size=(%s), stride=(%s)';
    my $len_kernel_size = @{ $self->kwargs->{kernel} };
    if(Dumper($self->kwargs->{pad}) ne Dumper([(0)x$len_kernel_size]))
    {
        $s .= ', padding=(' . join(',', @{ $self->kwargs->{pad} }) . ')';
    }
    if(Dumper($self->kwargs->{dilate}) ne Dumper([(1)x$len_kernel_size]))
    {
        $s .= ', dilation=(' . join(',', @{ $self->kwargs->{dilate} }) . ')';
    }
    if($self->can('out_pad') and Dumper($self->out_pad) ne Dumper([(0)x$len_kernel_size]))
    {
        $s .= ', output_padding=(' . join(',', @{ $self->kwargs->{dilate} }) . ')';
    }
    if($self->kwargs->{num_group} != 1)
    {
        $s .= ', groups=' . $self->kwargs->{num_group};
    }
    if(not defined $self->bias)
    {
        $s .= ', bias=False';
    }
    $s .= ')';
    return sprintf(
        $s,
        $self->_class_name,
        $self->in_channels
            ? sprintf("%d -> %d", $self->in_channels, $self->channels)
            : sprintf("%d", $self->channels),
        join(',', @{ $self->kwargs->{kernel} }),
        join(',', @{ $self->kwargs->{stride} })
    );
};

package AI::MXNet::Gluon::NN::Conv1D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv1D
=cut

=head1 DESCRIPTION

    1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 1 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 1 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 1 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 1 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout: str, default 'NCW'
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', etc.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 3D array of shape
        (batch_size, in_channels, width) if `layout` is `NCW`.

    Output shape:
        This depends on the `layout` parameter. Output is 3D array of shape
        (batch_size, channels, out_width) if `layout` is `NCW`.
        out_width is calculated as::

            out_width = floor((width+2*padding-dilation*(kernel_size-1)-1)/stride)+1
=cut

has '+strides'    => (default => 1);
has '+padding'    => (default => 0);
has '+dilation'   => (default => 1);
has '+groups'     => (default => 1);
has '+layout'     => (default => 'NCW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([$self->kernel_size]);
    }
    confess("kernel_size must be a number or an array ref of 1 ints")
        unless @{ $self->kernel_size } == 1;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Conv2D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv2D
=cut

=head1 DESCRIPTION

    2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 2 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 2 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 2 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 4D array of shape
        (batch_size, in_channels, height, width) if `layout` is `NCHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 4D array of shape
        (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

        out_height and out_width are calculated as::

            out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
            out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
=cut

has '+strides'    => (default => sub { [1, 1] });
has '+padding'    => (default => sub { [0, 0] });
has '+dilation'   => (default => sub { [1, 1] });
has '+groups'     => (default => 1);
has '+layout'     => (default => 'NCHW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([($self->kernel_size)x2]);
    }
    confess("kernel_size must be a number or an array ref of 2 ints")
        unless @{ $self->kernel_size } == 2;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Conv3D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv3D
=cut

=head1 DESCRIPTION

    3D convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is `True`,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 3 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. Convolution is applied on the 'D',
        'H' and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 5D array of shape
        (batch_size, in_channels, depth, height, width) if `layout` is `NCDHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 5D array of shape
        (batch_size, channels, out_depth, out_height, out_width) if `layout` is
        `NCDHW`.

        out_depth, out_height and out_width are calculated as::

            out_depth = floor((depth+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
            out_height = floor((height+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
            out_width = floor((width+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1
=cut

has '+strides'    => (default => sub { [1, 1, 1] });
has '+padding'    => (default => sub { [0, 0, 0] });
has '+dilation'   => (default => sub { [1, 1, 1] });
has '+groups'     => (default => 1);
has '+layout'     => (default => 'NCDHW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([($self->kernel_size)x3]);
    }
    confess("kernel_size must be a number or an array ref of 3 ints")
        unless @{ $self->kernel_size } == 3;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Conv1DTranspose;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv1DTranspose
=cut

=head1 DESCRIPTION

    Transposed 1D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 3 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCW'
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', etc.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Convolution is applied on the 'W' dimension.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 3D array of shape
        (batch_size, in_channels, width) if `layout` is `NCW`.

    Output shape:
        This depends on the `layout` parameter. Output is 3D array of shape
        (batch_size, channels, out_width) if `layout` is `NCW`.

        out_width is calculated as::

            out_width = (width-1)*strides-2*padding+kernel_size+output_padding
=cut

has 'output_padding' => (is => 'rw', isa => 'DimSize|Shape', default => 0);
has '+adj'           => (default => sub { shift->output_padding }, lazy => 1);
has '+op_name'       => (default => 'Deconvolution');
has '+strides'       => (default => 1);
has '+padding'       => (default => 0 );
has '+dilation'      => (default => 1);
has '+groups'        => (default => 1);
has '+layout'        => (default => 'NCW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([$self->kernel_size]);
    }
    if(not ref $self->output_padding)
    {
        $self->output_padding([$self->output_padding]);
    }
    confess("kernel_size must be a number or an array ref of 1 ints")
        unless @{ $self->kernel_size } == 1;
    confess("output_padding must be a number or an array ref of 1 ints")
        unless @{ $self->output_padding } == 1;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Conv2DTranspose;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv2DTranspose
=cut

=head1 DESCRIPTION

    Transposed 2D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 3 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 4D array of shape
        (batch_size, in_channels, height, width) if `layout` is `NCHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 4D array of shape
        (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

        out_height and out_width are calculated as::

            out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
            out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
=cut

has 'output_padding'      => (is => 'rw', isa => 'DimSize|Shape', default => 0);
has '+adj'        => (default => sub { shift->output_padding }, lazy => 1);
has '+op_name'    => (default => 'Deconvolution');
has '+strides'    => (default => sub { [1, 1] });
has '+padding'    => (default => sub { [0, 0] });
has '+dilation'   => (default => sub { [1, 1] });
has '+groups'     => (default => 1);
has '+layout'     => (default => 'NCHW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([($self->kernel_size)x2]);
    }
    if(not ref $self->output_padding)
    {
        $self->output_padding([($self->output_padding)x2]);
    }
    confess("kernel_size must be a number or an array ref of 2 ints")
        unless @{ $self->kernel_size } == 2;
    confess("output_padding must be a number or an array ref of 2 ints")
        unless @{ $self->output_padding } == 2;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::Conv3DTranspose;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Conv';

=head1 NAME

    AI::MXNet::Gluon::NN::Conv3DTranspose
=cut

=head1 DESCRIPTION

    Transposed 3D convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.


    Parameters
    ----------
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 3 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. Convolution is applied on the 'D',
        'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Input shape:
        This depends on the `layout` parameter. Input is 5D array of shape
        (batch_size, in_channels, depth, height, width) if `layout` is `NCDHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 5D array of shape
        (batch_size, channels, out_depth, out_height, out_width) if `layout` is `NCDHW`.
        out_depth, out_height and out_width are calculated as::

            out_depth = (depth-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
            out_height = (height-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
            out_width = (width-1)*strides[2]-2*padding[2]+kernel_size[2]+output_padding[2]
=cut

has 'output_padding'      => (is => 'rw', isa => 'DimSize|Shape', default => 0);
has '+adj'        => (default => sub { shift->output_padding }, lazy => 1);
has '+op_name'    => (default => 'Deconvolution');
has '+strides'    => (default => sub { [1, 1, 1] });
has '+padding'    => (default => sub { [0, 0, 0] });
has '+dilation'   => (default => sub { [1, 1, 1] });
has '+groups'     => (default => 1);
has '+layout'     => (default => 'NCDHW');

method _update_kernel_size()
{
    if(not ref $self->kernel_size)
    {
        $self->kernel_size([($self->kernel_size)x3]);
    }
    if(not ref $self->output_padding)
    {
        $self->output_padding([($self->output_padding)x3]);
    }
    confess("kernel_size must be a number or an array ref of 3 ints")
        unless @{ $self->kernel_size } == 3;
    confess("output_padding must be a number or an array ref of 3 ints")
        unless @{ $self->output_padding } == 3;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

# Abstract class for different pooling layers.
package AI::MXNet::Gluon::NN::Pooling;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

has 'pool_size'   => (is => 'rw', isa => 'DimSize|Shape', required => 1);
has 'strides'     => (is => 'rw', isa => 'Maybe[DimSize|Shape]');
has 'padding'     => (is => 'rw', isa => 'DimSize|Shape');
has 'ceil_mode'   => (is => 'rw', isa => 'Bool', default => 0);
has 'global_pool' => (is => 'rw', isa => 'Bool', default => 0);
has 'kwargs'      => (is => 'rw', init_arg => undef);
has 'pool_type'   => (is => 'rw', isa => 'PoolType');
has 'layout'      => (is => 'rw');
method python_constructor_arguments() { [qw/pool_size strides padding/] }

sub BUILD
{
    my $self = shift;
    $self->_update_pool_size;
    if(not defined $self->strides)
    {
        $self->strides($self->pool_size);
    }
    if(not ref $self->strides)
    {
        $self->strides([($self->strides)x@{ $self->pool_size }]);
    }
    if(not ref $self->padding)
    {
        $self->padding([($self->padding)x@{ $self->pool_size }]);
    }
    $self->kwargs({
        kernel => $self->pool_size, stride => $self->strides, pad => $self->padding,
        global_pool => $self->global_pool, pool_type => $self->pool_type,
        pooling_convention => $self->ceil_mode ? 'full' : 'valid'
    });
}

method _alias() { 'pool' }

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->Pooling($x, name=>'fwd', %{ $self->kwargs });
}

use overload '""' => sub {
    my $self = shift;
    sprintf(
        '%s(size=(%s), stride=(%s), padding=(%s), ceil_mode=%d)',
        $self->_class_name,
        join(',', @{ $self->kwargs->{kernel} }),
        join(',', @{ $self->kwargs->{stride} }),
        join(',', @{ $self->kwargs->{pad} }),
        $self->kwargs->{pooling_convention} eq 'full' ? 1 : 0
    )
};

package AI::MXNet::Gluon::NN::MaxPool1D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Pooling';
method python_constructor_arguments() { [qw/pool_size strides padding layout ceil_mode/] }

=head1 NAME

    AI::MXNet::Gluon::NN::MaxPool1D
=cut

=head1 DESCRIPTION

    Max pooling operation for one dimensional data.


    Parameters
    ----------
    pool_size: int
        Size of the max pooling windows.
    strides: int, or None
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCW'
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', etc.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. Pooling is applied on the W dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 3D array of shape
        (batch_size, channels, width) if `layout` is `NCW`.

    Output shape:
        This depends on the `layout` parameter. Output is 3D array of shape
        (batch_size, channels, out_width) if `layout` is `NCW`.

        out_width is calculated as::

            out_width = floor((width+2*padding-pool_size)/strides)+1

        When `ceil_mode` is `True`, ceil will be used instead of floor in this
        equation.
=cut


has '+pool_size' => (default => 2);
has '+padding'   => (default => 0);
has '+layout'    => (default => 'NCW');
has '+pool_type' => (default => 'max');

method _update_pool_size()
{
    confess("Only supports NCW layout for now")
        unless $self->layout eq 'NCW';
    if(not ref $self->pool_size)
    {
        $self->pool_size([$self->pool_size]);
    }
    confess("pool_size must be a number or an array ref of 1 ints")
        unless @{ $self->pool_size } == 1;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::MaxPool2D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Pooling';

=head1 NAME

    AI::MXNet::Gluon::NN::MaxPool2D
=cut

=head1 DESCRIPTION

    Max pooling operation for two dimensional (spatial) data.


    Parameters
    ----------
    pool_size: int or list/tuple of 2 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 2 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 2 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 4D array of shape
        (batch_size, channels, height, width) if `layout` is `NCHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 4D array of shape
        (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.

        out_height and out_width are calculated as::

            out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
            out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

        When `ceil_mode` is `True`, ceil will be used instead of floor in this
        equation.
=cut

has '+pool_size' => (default => sub { [2, 2] });
has '+padding'   => (default => 0);
has '+layout'    => (default => 'NCHW');
has '+pool_type' => (default => 'max');

method _update_pool_size()
{
    confess("Only supports NCHW layout for now")
        unless $self->layout eq 'NCHW';
    if(not ref $self->pool_size)
    {
        $self->pool_size([($self->pool_size)x2]);
    }
    confess("pool_size must be a number or an array ref of 2 ints")
        unless @{ $self->pool_size } == 2;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::MaxPool3D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::Pooling';

=head1 NAME

    AI::MXNet::Gluon::NN::MaxPool3D
=cut

=head1 DESCRIPTION

    Max pooling operation for 3D data (spatial or spatio-temporal).


    Parameters
    ----------
    pool_size: int or list/tuple of 3 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 3 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 3 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 5D array of shape
        (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 5D array of shape
        (batch_size, channels, out_depth, out_height, out_width) if `layout`
        is `NCDHW`.

        out_depth, out_height and out_width are calculated as ::

            out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
            out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
            out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

        When `ceil_mode` is `True`, ceil will be used instead of floor in this
        equation.
=cut

has '+pool_size' => (default => sub { [2, 2, 2] });
has '+padding'   => (default => 0);
has '+layout'    => (default => 'NCDHW');
has '+pool_type' => (default => 'max');

method _update_pool_size()
{
    confess("Only supports NCDHW layout for now")
        unless $self->layout eq 'NCDHW';
    if(not ref $self->pool_size)
    {
        $self->pool_size([($self->pool_size)x3]);
    }
    confess("pool_size must be a number or an array ref of 3 ints")
        unless @{ $self->pool_size } == 3;
}

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::AvgPool1D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool1D';

=head1 NAME

    AI::MXNet::Gluon::NN::AvgPool1D
=cut

=head1 DESCRIPTION

    Average pooling operation for temporal data.

    Parameters
    ----------
    pool_size: int
        Size of the max pooling windows.
    strides: int, or None
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCW'
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', etc.
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on 'W' dimension.
    ceil_mode : bool, default False
        When `True`, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 3D array of shape
        (batch_size, channels, width) if `layout` is `NCW`.

    Output shape:
        This depends on the `layout` parameter. Output is 3D array of shape
        (batch_size, channels, out_width) if `layout` is `NCW`.

        out_width is calculated as::

            out_width = floor((width+2*padding-pool_size)/strides)+1

        When `ceil_mode` is `True`, ceil will be used instead of floor in this
        equation.
=cut

has '+pool_type' => (default => 'avg');

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::AvgPool2D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool2D';

=head1 NAME

    AI::MXNet::Gluon::NN::AvgPool2D
=cut

=head1 DESCRIPTION

    Average pooling operation for spatial data.

    Parameters
    ----------
    pool_size: int or list/tuple of 2 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 2 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 2 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCHW'
        Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc.
        'N', 'C', 'H', 'W' stands for batch, channel, height, and width
        dimensions respectively. padding is applied on 'H' and 'W' dimension.
    ceil_mode : bool, default False
        When True, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 4D array of shape
        (batch_size, channels, height, width) if `layout` is `NCHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 4D array of shape
        (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.

        out_height and out_width are calculated as::

            out_height = floor((height+2*padding[0]-pool_size[0])/strides[0])+1
            out_width = floor((width+2*padding[1]-pool_size[1])/strides[1])+1

        When `ceil_mode` is `True`, ceil will be used instead of floor in this
        equation.
=cut

has '+pool_type' => (default => 'avg');

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::AvgPool3D;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool3D';

=head1 NAME

    AI::MXNet::Gluon::NN::AvgPool3D
=cut

=head1 DESCRIPTION

    Average pooling operation for 3D data (spatial or spatio-temporal).

    Parameters
    ----------
    pool_size: int or list/tuple of 3 ints,
        Size of the max pooling windows.
    strides: int, list/tuple of 3 ints, or None.
        Factor by which to downscale. E.g. 2 will halve the input size.
        If `None`, it will default to `pool_size`.
    padding: int or list/tuple of 3 ints,
        If padding is non-zero, then the input is implicitly
        zero-padded on both sides for padding number of points.
    layout : str, default 'NCDHW'
        Dimension ordering of data and weight. Can be 'NCDHW', 'NDHWC', etc.
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.
    ceil_mode : bool, default False
        When True, will use ceil instead of floor to compute the output shape.


    Input shape:
        This depends on the `layout` parameter. Input is 5D array of shape
        (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.

    Output shape:
        This depends on the `layout` parameter. Output is 5D array of shape
        (batch_size, channels, out_depth, out_height, out_width) if `layout`
        is `NCDHW`.

        out_depth, out_height and out_width are calculated as ::

            out_depth = floor((depth+2*padding[0]-pool_size[0])/strides[0])+1
            out_height = floor((height+2*padding[1]-pool_size[1])/strides[1])+1
            out_width = floor((width+2*padding[2]-pool_size[2])/strides[2])+1

        When `ceil_mode` is `True,` ceil will be used instead of floor in this
        equation.
=cut

has '+pool_type' => (default => 'avg');
__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalMaxPool1D;

=head1 NAME

    AI::MXNet::Gluon::NN::GlobalMaxPool1D
=cut

=head1 DESCRIPTION

    Global max pooling operation for temporal data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool1D';
has '+pool_size'   => (default => sub { [1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalMaxPool2D;
=head1 NAME

    AI::MXNet::Gluon::NN::GlobalMaxPool2D
=cut

=head1 DESCRIPTION

    Global max pooling operation for spatial data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool2D';

has '+pool_size'   => (default => sub { [1, 1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalMaxPool3D;
=head1 NAME

    AI::MXNet::Gluon::NN::GlobalMaxPool3D
=cut

=head1 DESCRIPTION

    Global max pooling operation for 3D data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::MaxPool3D';
has '+pool_size'   => (default => sub { [1, 1, 1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalAvgPool1D;

=head1 NAME

    AI::MXNet::Gluon::NN::GlobalAvgPool1D
=cut

=head1 DESCRIPTION

    Global average pooling operation for temporal data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::AvgPool1D';
has '+pool_size'   => (default => sub { [1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalAvgPool2D;
=head1 NAME

    AI::MXNet::Gluon::NN::GlobalAvgPool2D
=cut

=head1 DESCRIPTION

    Global average pooling operation for spatial data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::AvgPool2D';

has '+pool_size'   => (default => sub { [1, 1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

package AI::MXNet::Gluon::NN::GlobalAvgPool3D;
=head1 NAME

    AI::MXNet::Gluon::NN::GlobalAvgPool2D
=cut

=head1 DESCRIPTION

    Global average pooling operation for 3D data.
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::NN::AvgPool3D';
has '+pool_size'   => (default => sub { [1, 1, 1] });
has '+global_pool' => (default => 1);
has '+ceil_mode'   => (default => 1);

__PACKAGE__->register('AI::MXNet::Gluon::NN');

1;