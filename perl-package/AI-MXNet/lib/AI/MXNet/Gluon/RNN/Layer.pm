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
package AI::MXNet::Gluon::RNN::Layer;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::Block';

has 'hidden_size'   => (is => 'rw', isa => 'Int');
has 'num_layers'    => (is => 'rw', isa => 'Int');
has 'layout'        => (is => 'rw', isa => 'Str');
has 'dropout'       => (is => 'rw', isa => 'Num');
has 'bidirectional' => (is => 'rw', isa => 'Bool');
has 'input_size'    => (is => 'rw', isa => 'Int', default => 0);
has [qw/
    i2h_weight_initializer
    h2h_weight_initializer
    i2h_bias_initializer
    h2h_bias_initializer
    /]              => (is => 'rw', isa => 'Maybe[Initializer]');
has 'mode'          => (is => 'rw', isa => 'Str');
has [qw/dir gates
    i2h_weight
    h2h_weight
    i2h_bias
    h2h_bias
    unfused/]       => (is => 'rw', init_arg => undef);

method python_constructor_arguments()
{
    [qw/
        hidden_size num_layers layout
        dropout bidirectional input_size
        i2h_weight_initializer h2h_weight_initializer
        i2h_bias_initializer h2h_bias_initializer
        mode
    /];
}

sub BUILD
{
    my $self = shift;
    assert(
        ($self->layout eq 'TNC' or $self->layout eq 'NTC'),
        "Invalid layout [${\ $self->layout }]; must be one of ['TNC' or 'NTC']"
    );
    $self->i2h_weight([]);
    $self->h2h_weight([]);
    $self->i2h_bias([]);
    $self->h2h_bias([]);
    $self->dir($self->bidirectional ? 2 : 1);
    $self->gates({qw/rnn_relu 1 rnn_tanh 1 lstm 4 gru 3/}->{$self->mode});
    my ($ng, $ni, $nh) = ($self->gates, $self->input_size, $self->hidden_size);
    for my $i (0..$self->num_layers-1)
    {
        for my $j ($self->dir == 2 ? ('l', 'r') : ('l'))
        {
            push @{ $self->i2h_weight }, $self->params->get(
                "$j${i}_i2h_weight", shape=>[$ng*$nh, $ni],
                init => $self->i2h_weight_initializer,
                allow_deferred_init => 1
            );
            push @{ $self->h2h_weight }, $self->params->get(
                "$j${i}_h2h_weight", shape=>[$ng*$nh, $nh],
                init => $self->h2h_weight_initializer,
                allow_deferred_init => 1
            );
            push @{ $self->i2h_bias }, $self->params->get(
                "$j${i}_i2h_bias", shape=>[$ng*$nh],
                init => $self->i2h_bias_initializer,
                allow_deferred_init => 1
            );
            push @{ $self->h2h_bias }, $self->params->get(
                "$j${i}_h2h_bias", shape=>[$ng*$nh],
                init => $self->h2h_bias_initializer,
                allow_deferred_init => 1
            );
        }
        $ni = $nh * $self->dir;
    }
    $self->unfused($self->_unfuse());
}

use overload '""' => sub {
    my $self = shift;
    my $name = $self->_class_name;
    my $mapping = $self->input_size ? $self->input_size.' -> '.$self->hidden_size : $self->hidden_size;
    my $s = "$name($mapping, ${\ $self->layout }";
    if($self->num_layers != 1)
    {
        $s .= ', num_layers='.$self->num_layers;
    }
    if($self->dropout != 0)
    {
        $s .= ', dropout='.$self->dropout;
    }
    if($self->dir == 2)
    {
        $s .= ', bidirectional';
    }
    $s .= ')';
    return $s;
};

method state_info($batch_size=0)
{
    confess('NotImplementedError');
}

# Unfuses the fused RNN in to a stack of rnn cells.

method _unfuse()
{
    my $get_cell = {
        rnn_relu => sub {
            my %kwargs = @_;
            AI::MXNet::Gluon::RNN::RNNCell->new(
                $self->hidden_size,
                activation => 'relu',
                %kwargs
            )
        },
        rnn_tanh => sub {
            my %kwargs = @_;
            AI::MXNet::Gluon::RNN::RNNCell->new(
                $self->hidden_size,
                activation => 'tanh',
                %kwargs
            )
        },
        lstm => sub {
            my %kwargs = @_;
            AI::MXNet::Gluon::RNN::LSTMCell->new(
                $self->hidden_size,
                %kwargs
            )
        },
        gru => sub {
            my %kwargs = @_;
            AI::MXNet::Gluon::RNN::GRUCell->new(
                $self->hidden_size,
                %kwargs
            )
        }
    }->{$self->mode};
    my $stack = AI::MXNet::Gluon::RNN::SequentialRNNCell->new(prefix => $self->prefix, params => $self->params);
    $stack->name_scope(sub {
        my $ni = $self->input_size;
        for my $i (0..$self->num_layers-1)
        {
            my %kwargs = (
                input_size => $ni,
                i2h_weight_initializer => $self->i2h_weight_initializer,
                h2h_weight_initializer => $self->h2h_weight_initializer,
                i2h_bias_initializer   => $self->i2h_bias_initializer,
                h2h_bias_initializer   => $self->h2h_bias_initializer
            );
            if($self->dir == 2)
            {
                $stack->add(
                    AI::MXNet::Gluon::RNN::BidirectionalCell->new(
                        $get_cell->(prefix=> "l${i}_", %kwargs),
                        $get_cell->(prefix=> "r${i}_", %kwargs),
                    )
                );
            }
            else
            {
                $stack->add($get_cell->(prefix=> "l${i}_", %kwargs));
            }
            if($self->dropout > 0 and $i != ($self->_num_layers - 1))
            {
                $stack->add(AI::MXNet::Gluon::RNN::DropoutCell->new($self->dropout));
            }
            $ni = $self->hidden_size * $self->dir;
        }
    });
    return $stack;
}

method begin_state(
    $batch_size=0,
    CodeRef :$func=sub { my %kwargs = @_; my $shape = delete $kwargs{shape}; AI::MXNet::NDArray->zeros($shape, %kwargs) },
    %kwargs
)
{
    my @states;
    enumerate(sub {
        my ($i, $info) = @_;
        if(defined $info)
        {
            %$info = (%$info, %kwargs);
        }
        else
        {
            %$info = %kwargs;
        }
        push @states, $func->(name=> $self->prefix."h0_$i", %$info);
    }, $self->state_info($batch_size));
    return \@states;
}

use Data::Dumper;
method forward(GluonInput $inputs, Maybe[GluonInput] $states=)
{
    my $batch_size = $inputs->shape->[index($self->layout, 'N')];
    my $skip_states = not defined $states;
    if($skip_states)
    {
        $states = $self->begin_state($batch_size, ctx=>$inputs->context);
    }
    if(blessed $states and $states->isa('AI::MXNet::NDArray'))
    {
        $states = [$states];
    }
    for(zip($states, $self->state_info($batch_size))) {
        my ($state, $info) = @$_;
        if(Dumper($state->shape) ne Dumper($info->{shape}))
        {
            my @state_shape = @{ $state->shape };
            confess("Invalid recurrent state shape. Expecting @{$info->{shape}}, got @state_shape.");
        }
    }
    if($self->input_size == 0)
    {
        for my $i (0..$self->dir-1)
        {
            $self->i2h_weight->[$i]->shape([$self->gates*$self->hidden_size, $inputs->shape->[2]]);
            $self->i2h_weight->[$i]->_finish_deferred_init();
        }
    }
    my $out;
    if($inputs->context->device_type eq 'gpu')
    {
        $out = $self->_forward_gpu($inputs, $states);
    }
    else
    {
        $out = $self->_forward_cpu($inputs, $states);
    }

    # out is (output, state)
    return $skip_states ? $out->[0] : $out;
}

method _forward_cpu($inputs, $states)
{
    my $ns = @{ $states };
    my $axis = index($self->layout, 'T');
    $states = [map { @{$_} } @{ $states }];
    my $outputs;
    ($outputs, $states) = $self->unfused->unroll(
        $inputs->shape->[$axis], $inputs, begin_state => $states,
        layout => $self->layout, merge_outputs => 1
    );
    my @new_states;
    for my $i (0..$ns-1)
    {
        my @tmp;
        for (my $j = $i; $j < @{ $states }; $j += $ns)
        {
            push @tmp, $states->[$j];
        }
        my $state = AI::MXNet::NDArray->concat((map { $_->reshape([1, @{ $_->shape }]) } @tmp), dim => 0);
        push @new_states, $state;
    }
    return [$outputs, \@new_states];
}

method _forward_gpu($inputs, $states)
{
    if($self->layout eq 'NTC')
    {
        $inputs = $inputs->swapaxes(dim1 => 0, dim2 => 1);
    }
    my $ctx = $inputs->context;
    my @params = map { $_->data($ctx)->reshape([-1]) } map { @{ $_ } } (
        $self->i2h_weight, $self->h2h_weight,
        $self->i2h_bias, $self->h2h_bias
    );
    my $params = AI::MXNet::NDArray->concat(@params, dim => 0);
    my $rnn = AI::MXNet::NDArray->RNN(
        $inputs, $params, @{ $states }, state_size => $self->hidden_size,
        num_layers => $self->num_layers, bidirectional => $self->dir == 2 ? 1 : 0,
        p => $self->dropout, state_outputs => 1, mode => $self->mode
    );
    my $outputs;
    my @rnn = @{$rnn};
    if($self->mode eq 'lstm')
    {
        ($outputs, $states) = ($rnn[0], [$rnn[1], $rnn[2]]);
    }
    else
    {
        ($outputs, $states) = ($rnn[0], [$rnn[1]]);
    }
    if($self->layout eq 'NTC')
    {
        $outputs = $outputs->swapaxes(dim1 => 0, dim2 => 1);
    }
    return [$outputs, $states];
}


package AI::MXNet::Gluon::RNN::RNN;

=head1 NAME

     AI::MXNet::Gluon::RNN::RNN
=cut

=head1 DESCRIPTION

    Applies a multi-layer Elman RNN with `tanh` or `ReLU` non-linearity to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first layer.
    If nonlinearity='relu', then `ReLU` is used instead of `tanh`.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h.
    num_layers: int, default 1
        Number of recurrent layers.
    activation: {'relu' or 'tanh'}, default 'tanh'
        The activation function to use.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    bidirectional: bool, default False
        If `True`, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    prefix : str or None
        Prefix of this `Block`.
    params : ParameterDict or None
        Shared Parameters for this `Block`.


    Input shapes:
        The input shape depends on `layout`. For `layout='TNC'`, the
        input has shape `(sequence_length, batch_size, input_size)`


    Output shape:
        The output shape depends on `layout`. For `layout='TNC'`, the
        output has shape `(sequence_length, batch_size, num_hidden)`.
        If `bidirectional` is True, output shape will instead be
        `(sequence_length, batch_size, 2*num_hidden)`

    Recurrent state:
        The recurrent state is an NDArray with shape `(num_layers, batch_size, num_hidden)`.
        If `bidirectional` is True, the recurrent state shape will instead be
        `(2*num_layers, batch_size, num_hidden)`
        If input recurrent state is None, zeros are used as default begin states,
        and the output recurrent state is omitted.


    Examples
    --------
    >>> layer = mx.gluon.rnn.RNN(100, 3)
    >>> layer.initialize()
    >>> input = mx.nd.random.uniform(shape=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.nd.random.uniform(shape=(3, 3, 100))
    >>> output, hn = layer(input, h0)
=cut
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::Layer';

has '+num_layers'    => (default => 1);
has 'activation'     => (is => 'rw', default => 'relu');
has '+layout'        => (default => 'TNC');
has '+dropout'       => (default => 0);
has '+bidirectional' => (default => 0);
has [qw/
    +i2h_bias_initializer
    +h2h_bias_initializer
    /]               => (default => 'zeros');
has '+mode'          => (default => sub { 'rnn_' . shift->activation }, lazy => 1);
method python_constructor_arguments()
{
    [qw/
        hidden_size num_layers activation layout
        dropout bidirectional input_size
        i2h_weight_initializer h2h_weight_initializer
        i2h_bias_initializer h2h_bias_initializer
    /];
}

method state_info(DimSize $batch_size=0)
{
    return [{
        shape => [$self->num_layers * $self->dir, $batch_size, $self->hidden_size],
        __layout__ => 'LNC'
    }];
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::LSTM;

=head1 NANE

    AI::MXNet::Gluon::RNN::LSTM
=cut

=head1 DESCRIPTION

    Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
        o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        h_t = o_t * \tanh(c_t)
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the hidden state of the previous
    layer at time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h.
    num_layers: int, default 1
        Number of recurrent layers.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    bidirectional: bool, default False
        If `True`, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'lstmbias'
        Initializer for the bias vector. By default, bias for the forget
        gate is initialized to 1 while all other biases are initialized
        to zero.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    prefix : str or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.


    Input shapes:
        The input shape depends on `layout`. For `layout='TNC'`, the
        input has shape `(sequence_length, batch_size, input_size)`

    Output shape:
        The output shape depends on `layout`. For `layout='TNC'`, the
        output has shape `(sequence_length, batch_size, num_hidden)`.
        If `bidirectional` is True, output shape will instead be
        `(sequence_length, batch_size, 2*num_hidden)`

    Recurrent state:
        The recurrent state is a list of two NDArrays. Both has shape
        `(num_layers, batch_size, num_hidden)`.
        If `bidirectional` is True, each recurrent state will instead have shape
        `(2*num_layers, batch_size, num_hidden)`.
        If input recurrent state is None, zeros are used as default begin states,
        and the output recurrent state is omitted.


    Examples
    --------
    >>> layer = mx.gluon.rnn.LSTM(100, 3)
    >>> layer.initialize()
    >>> input = mx.nd.random.uniform(shape=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.nd.random.uniform(shape=(3, 3, 100))
    >>> c0 = mx.nd.random.uniform(shape=(3, 3, 100))
    >>> output, hn = layer(input, [h0, c0])
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::Layer';

has '+num_layers'    => (default => 1);
has '+layout'        => (default => 'TNC');
has '+dropout'       => (default => 0);
has '+bidirectional' => (default => 0);
has [qw/
    +i2h_bias_initializer
    +h2h_bias_initializer
    /]               => (default => 'zeros');
has '+mode'          => (default => 'lstm');

method state_info(DimSize $batch_size=0)
{
    return [
        {
            shape => [$self->num_layers * $self->dir, $batch_size, $self->hidden_size],
            __layout__ => 'LNC'
        },
        {
            shape => [$self->num_layers * $self->dir, $batch_size, $self->hidden_size],
            __layout__ => 'LNC'
        }
    ];
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::GRU;

=head1 NANE

    AI::MXNet::Gluon::RNN::GRU
=cut

=head1 DESCRIPTION

    Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
        r_t = sigmoid(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_hi h_{(t-1)} + b_{hi}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - i_t) * n_t + i_t * h_{(t-1)} \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first layer,
    and :math:`r_t`, :math:`i_t`, :math:`n_t` are the reset, input, and new gates, respectively.

    Parameters
    ----------
    hidden_size: int
        The number of features in the hidden state h
    num_layers: int, default 1
        Number of recurrent layers.
    layout : str, default 'TNC'
        The format of input and output tensors. T, N and C stand for
        sequence length, batch size, and feature dimensions respectively.
    dropout: float, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer
    bidirectional: bool, default False
        If True, becomes a bidirectional RNN.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    input_size: int, default 0
        The number of expected features in the input x.
        If not specified, it will be inferred from input.
    prefix : str or None
        Prefix of this `Block`.
    params : ParameterDict or None
        Shared Parameters for this `Block`.


    Input shapes:
        The input shape depends on `layout`. For `layout='TNC'`, the
        input has shape `(sequence_length, batch_size, input_size)`

    Output shape:
        The output shape depends on `layout`. For `layout='TNC'`, the
        output has shape `(sequence_length, batch_size, num_hidden)`.
        If `bidirectional` is True, output shape will instead be
        `(sequence_length, batch_size, 2*num_hidden)`

    Recurrent state:
        The recurrent state is an NDArray with shape `(num_layers, batch_size, num_hidden)`.
        If `bidirectional` is True, the recurrent state shape will instead be
        `(2*num_layers, batch_size, num_hidden)`
        If input recurrent state is None, zeros are used as default begin states,
        and the output recurrent state is omitted.


    Examples
    --------
    >>> layer = mx.gluon.rnn.GRU(100, 3)
    >>> layer.initialize()
    >>> input = mx.nd.random.uniform(shape=(5, 3, 10))
    >>> # by default zeros are used as begin state
    >>> output = layer(input)
    >>> # manually specify begin state.
    >>> h0 = mx.nd.random.uniform(shape=(3, 3, 100))
    >>> output, hn = layer(input, h0)
=cut

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::Layer';

has '+num_layers'    => (default => 1);
has '+layout'        => (default => 'TNC');
has '+dropout'       => (default => 0);
has '+bidirectional' => (default => 0);
has [qw/
    +i2h_bias_initializer
    +h2h_bias_initializer
    /]               => (default => 'zeros');
has '+mode'          => (default => 'gru');

method state_info(DimSize $batch_size=0)
{
    return [
        {
            shape => [$self->num_layers * $self->dir, $batch_size, $self->hidden_size],
            __layout__ => 'LNC'
        }
    ];
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

1;
