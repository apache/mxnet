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
package AI::MXNet::Gluon::RNN::RecurrentCell;
use Mouse::Role;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

method _cells_state_info($cells, $batch_size)
{
    return [map { @{ $_->state_info($batch_size) } } @{ $cells }];
}

method _cells_begin_state($cells, %kwargs)
{
    return [map { @{ $_->begin_state(%kwargs) } } @{ $cells }];
}

method _get_begin_state(GluonClass $F, $begin_state, GluonInput $inputs, $batch_size)
{
    if(not defined $begin_state)
    {
        if($F =~ /AI::MXNet::NDArray/)
        {
            my $ctx = blessed $inputs ? $inputs->context : $inputs->[0]->context;
            {
                local($AI::MXNet::current_ctx) = $ctx;
                my $func = sub {
                    my %kwargs = @_;
                    my $shape = delete $kwargs{shape};
                    return AI::MXNet::NDArray->zeros($shape, %kwargs);
                };
                $begin_state = $self->begin_state(batch_size => $batch_size, func => $func);
            }
        }
        else
        {
            $begin_state = $self->begin_state(batch_size => $batch_size, func => sub { return $F->zeros(@_) });
        }
    }
    return $begin_state;
}

method _format_sequence($length, $inputs, $layout, $merge, $in_layout=)
{
    assert(
        (defined $inputs),
        "unroll(inputs=None) has been deprecated. ".
        "Please create input variables outside unroll."
    );

    my $axis = index($layout, 'T');
    my $batch_axis = index($layout, 'N');
    my $batch_size = 0;
    my $in_axis = defined $in_layout ? index($in_layout, 'T') : $axis;
    my $F;
    if(blessed $inputs and $inputs->isa('AI::MXNet::Symbol'))
    {
        $F = 'AI::MXNet::Symbol';
        if(not $merge)
        {
            assert(
                (@{ $inputs->list_outputs() } == 1),
                "unroll doesn't allow grouped symbol as input. Please convert ".
                "to list with list(inputs) first or let unroll handle splitting"
            );
            $inputs = [
                AI::MXNet::Symbol->split(
                    $inputs, axis => $in_axis, num_outputs => $length, squeeze_axis => 1
                )
            ];
        }
    }
    elsif(blessed $inputs and $inputs->isa('AI::MXNet::NDArray'))
    {
        $F = 'AI::MXNet::NDArray';
        $batch_size = $inputs->shape->[$batch_axis];
        if(not $merge)
        {
            assert(not defined $length or $length == $inputs->shape->[$in_axis]);
            $inputs = as_array(
                AI::MXNet::NDArray->split(
                    $inputs, axis=>$in_axis,
                    num_outputs => $inputs->shape->[$in_axis],
                    squeeze_axis => 1
                )
            );
        }
    }
    else
    {
        assert(not defined $length or @{ $inputs } == $length);
        if($inputs->[0]->isa('AI::MXNet::Symbol'))
        {
            $F = 'AI::MXNet::Symbol';
        }
        else
        {
            $F = 'AI::MXNet::NDArray';
            $batch_size = $inputs->[0]->shape->[$batch_axis];
        }
        if($merge)
        {
            $inputs  = [map { $F->expand_dims($_, axis => $axis) } @{ $inputs }];
            $inputs  = $F->concat(@{ $inputs }, dim => $axis);
            $in_axis = $axis;
        }
    }
    if(blessed $inputs and $axis != $in_axis)
    {
        $inputs = $F->swapaxes($inputs, dim1=>$axis, dim2=>$in_axis);
    }
    return ($inputs, $axis, $F, $batch_size);
}

=head1 NAME

    AI::MXNet::Gluon::RNN::RecurrentCell
=cut

=head1 DESCRIPTION

    Abstract role for RNN cells

    Parameters
    ----------
    prefix : str, optional
        Prefix for names of `Block`s
        (this prefix is also used for names of weights if `params` is `None`
        i.e. if `params` are being created and not reused)
    params : Parameter or None, optional
        Container for weight sharing between cells.
        A new Parameter container is created if `params` is `None`.
=cut

=head2 reset

    Reset before re-using the cell for another graph.
=cut

method reset()
{
    $self->init_counter(-1);
    $self->counter(-1);
    $_->reset for @{ $self->_children };
}

=head2 state_info

    Shape and layout information of states
=cut
method state_info(Int $batch_size=0)
{
    confess('Not Implemented');
}

=head2 begin_state

        Initial state for this cell.

        Parameters
        ----------
        $func : CodeRef, default sub { AI::MXNet::Symbol->zeros(@_) }
            Function for creating initial state.

            For Symbol API, func can be `symbol.zeros`, `symbol.uniform`,
            `symbol.var etc`. Use `symbol.var` if you want to directly
            feed input as states.

            For NDArray API, func can be `ndarray.zeros`, `ndarray.ones`, etc.
        $batch_size: int, default 0
            Only required for NDArray API. Size of the batch ('N' in layout)
            dimension of input.

        %kwargs :
            Additional keyword arguments passed to func. For example
            `mean`, `std`, `dtype`, etc.

        Returns
        -------
        states : nested array ref of Symbol
            Starting states for the first RNN step.
=cut

method begin_state(Int :$batch_size=0, CodeRef :$func=, %kwargs)
{
    $func //= sub {
        my %kwargs = @_;
        my $shape = delete $kwargs{shape};
        return AI::MXNet::NDArray->zeros($shape, %kwargs);
    };
    assert(
        (not $self->modified),
        "After applying modifier cells (e.g. ZoneoutCell) the base ".
        "cell cannot be called directly. Call the modifier cell instead."
    );
    my @states;
    for my $info (@{ $self->state_info($batch_size) })
    {
        $self->init_counter($self->init_counter + 1);
        if(defined $info)
        {
            %$info = (%$info, %kwargs);
        }
        else
        {
            $info = \%kwargs;
        }
        my $state = $func->(
            name => "${\ $self->_prefix }begin_state_${\ $self->init_counter }",
            %$info
        );
        push @states, $state;
    }
    return \@states;
}

=head2 unroll

        Unrolls an RNN cell across time steps.

        Parameters
        ----------
        $length : int
            Number of steps to unroll.
        $inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if `layout` is 'NTC',
            or (length, batch_size, ...) if `layout` is 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        :$begin_state : nested list of Symbol, optional
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if `None`.
        :$layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        :$merge_outputs : bool, optional
            If `False`, returns outputs as a list of Symbols.
            If `True`, concatenates output across time steps
            and returns a single symbol with shape
            (batch_size, length, ...) if layout is 'NTC',
            or (length, batch_size, ...) if layout is 'TNC'.
            If `None`, output whatever is faster.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
=cut

method unroll(
    Int $length,
    Maybe[GluonInput] $inputs,
    Maybe[GluonInput] :$begin_state=,
    Str :$layout='NTC',
    Maybe[Bool] :$merge_outputs=
)
{
    $self->reset();
    my ($F, $batch_size);
    ($inputs, undef, $F, $batch_size) = $self->_format_sequence($length, $inputs, $layout, 0);
    $begin_state //= $self->_get_begin_state($F, $begin_state, $inputs, $batch_size);

    my $states = $begin_state;
    my $outputs = [];
    use Data::Dumper;
    for my $i (0..$length-1)
    {
        my $output;
        ($output, $states) = $self->($inputs->[$i], $states);
        push @$outputs, $output;
    }
    ($outputs) = $self->_format_sequence($length, $outputs, $layout, $merge_outputs);
    return ($outputs, $states);
}

method _get_activation(GluonClass $F, GluonInput $inputs, Activation $activation, %kwargs)
{
    if(not blessed $activation)
    {
        return $F->Activation($inputs, act_type=>$activation, %kwargs);
    }
    else
    {
        return $activation->($inputs, %kwargs);
    }
}

=head2 forward

        Unrolls the recurrent cell for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            Input symbol, 2D, of shape (batch_size * num_units).
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of `begin_state()`.
            This can be used as an input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
=cut

package AI::MXNet::Gluon::RNN::HybridRecurrentCell;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
with 'AI::MXNet::Gluon::RNN::RecurrentCell';
has 'modified'      => (is => 'rw', isa => 'Bool', default => 0);
has [qw/counter
     init_counter/] => (is => 'rw', isa => 'Int', default => -1);

sub BUILD
{
    my $self = shift;
    $self->reset;
}

use overload '""' => sub {
    my $self = shift;
    my $s = '%s(%s';
    if($self->can('activation'))
    {
        $s .= ", ${\ $self->activation }";
    }
    $s .= ')';
    my $mapping = $self->input_size ? $self->input_size . " -> " . $self->hidden_size : $self->hidden_size;
    return sprintf($s, $self->_class_name, $mapping);
};

method forward(GluonInput $inputs, Maybe[GluonInput|ArrayRef[GluonInput]] $states)
{
    $self->counter($self->counter + 1);
    $self->SUPER::forward($inputs, $states);
}

package AI::MXNet::Gluon::RNN::RNNCell;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';

=head1 NAME

    AI::MXNet::Gluon::RNN::RNNCell
=cut

=head1 DESCRIPTION

    Simple recurrent neural network cell.

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol
    activation : str or Symbol, default 'tanh'
        Type of activation function.
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
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
=cut

has 'hidden_size' => (is => 'rw', isa => 'Int', required => 1);
has 'activation'  => (is => 'rw', isa => 'Activation', default => 'tanh');
has [qw/
    i2h_weight_initializer
    h2h_weight_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]');
has [qw/
    i2h_bias_initializer
    h2h_bias_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]', default => 'zeros');
has 'input_size'  => (is => 'rw', isa => 'Int', default => 0);
has [qw/
        i2h_weight
        h2h_weight
        i2h_bias
        h2h_bias
    /]            => (is => 'rw', init_arg => undef);

method python_constructor_arguments()
{
    [qw/
        hidden_size activation 
        i2h_weight_initializer h2h_weight_initializer
        i2h_bias_initializer h2h_bias_initializer
        input_size
    /];
}

sub BUILD
{
    my $self = shift;
    $self->i2h_weight($self->params->get(
        'i2h_weight', shape=>[$self->hidden_size, $self->input_size],
        init => $self->i2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_weight($self->params->get(
        'h2h_weight', shape=>[$self->hidden_size, $self->hidden_size],
        init => $self->h2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->i2h_bias($self->params->get(
        'i2h_bias', shape=>[$self->hidden_size],
        init => $self->i2h_bias_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_bias($self->params->get(
        'h2h_bias', shape=>[$self->hidden_size],
        init => $self->h2h_bias_initializer,
        allow_deferred_init => 1
    ));
}

method state_info(Int $batch_size=0)
{
    return [{ shape => [$batch_size, $self->hidden_size], __layout__ => 'NC' }];
}

method _alias() { 'rnn' }

method hybrid_forward(
    GluonClass $F, GluonInput $inputs, GluonInput $states,
    GluonInput :$i2h_weight, GluonInput :$h2h_weight, GluonInput :$i2h_bias, GluonInput :$h2h_bias
)
{
    my $prefix = "t${\ $self->counter}_";
    my $i2h = $F->FullyConnected(
        $inputs, $i2h_weight, $i2h_bias,
        num_hidden => $self->hidden_size,
        name => "${prefix}i2h"
    );
    my $h2h = $F->FullyConnected(
        $states->[0], $h2h_weight, $h2h_bias,
        num_hidden => $self->hidden_size,
        name => "${prefix}h2h"
    );
    my $output = $self->_get_activation($F, $i2h + $h2h, $self->activation, name => "${prefix}out");
    return ($output, [$output]);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::LSTMCell;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';

=head1 NAME

    AI::MXNet::Gluon::RNN::LSTMCell
=cut

=head1 DESCRIPTION

    Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol.
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
    prefix : str, default 'lstm_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
=cut

has 'hidden_size' => (is => 'rw', isa => 'Int', required => 1);
has [qw/
    i2h_weight_initializer
    h2h_weight_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]');
has [qw/
    i2h_bias_initializer
    h2h_bias_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]', default => 'zeros');
has 'input_size'  => (is => 'rw', isa => 'Int', default => 0);
has [qw/
        i2h_weight
        h2h_weight
        i2h_bias
        h2h_bias
    /]            => (is => 'rw', init_arg => undef);

method python_constructor_arguments()
{
    [qw/
        hidden_size
        i2h_weight_initializer h2h_weight_initializer
        i2h_bias_initializer h2h_bias_initializer
        input_size
    /];
}

sub BUILD
{
    my $self = shift;
    $self->i2h_weight($self->params->get(
        'i2h_weight', shape=>[4*$self->hidden_size, $self->input_size],
        init => $self->i2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_weight($self->params->get(
        'h2h_weight', shape=>[4*$self->hidden_size, $self->hidden_size],
        init => $self->h2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->i2h_bias($self->params->get(
        'i2h_bias', shape=>[4*$self->hidden_size],
        init => $self->i2h_bias_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_bias($self->params->get(
        'h2h_bias', shape=>[4*$self->hidden_size],
        init => $self->h2h_bias_initializer,
        allow_deferred_init => 1
    ));
}

method state_info(Int $batch_size=0)
{
    return [
        { shape => [$batch_size, $self->hidden_size], __layout__ => 'NC' },
        { shape => [$batch_size, $self->hidden_size], __layout__ => 'NC' }
    ];
}

method _alias() { 'lstm' }

method hybrid_forward(
    GluonClass $F, GluonInput $inputs, GluonInput $states,
    GluonInput :$i2h_weight, GluonInput :$h2h_weight, GluonInput :$i2h_bias, GluonInput :$h2h_bias
)
{
    my $prefix = "t${\ $self->counter}_";
    my $i2h = $F->FullyConnected(
        $inputs, $i2h_weight, $i2h_bias,
        num_hidden => $self->hidden_size*4,
        name => "${prefix}i2h"
    );
    my $h2h = $F->FullyConnected(
        $states->[0], $h2h_weight, $h2h_bias,
        num_hidden => $self->hidden_size*4,
        name => "${prefix}h2h"
    );
    my $gates = $i2h + $h2h;
    my @slice_gates = @{ $F->SliceChannel($gates, num_outputs => 4, name => "${prefix}slice") };
    my $in_gate = $F->Activation($slice_gates[0], act_type=>"sigmoid", name => "${prefix}i");
    my $forget_gate = $F->Activation($slice_gates[1], act_type=>"sigmoid", name => "${prefix}f");
    my $in_transform = $F->Activation($slice_gates[2], act_type=>"tanh", name => "${prefix}c");
    my $out_gate = $F->Activation($slice_gates[3], act_type=>"sigmoid", name => "${prefix}o");
    my $next_c = $F->_plus($forget_gate * $states->[1], $in_gate * $in_transform, name => "${prefix}state");
    my $next_h = $F->_mul($out_gate, $F->Activation($next_c, act_type=>"tanh"), name => "${prefix}out");
    return ($next_h, [$next_h, $next_c]);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::GRUCell;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';

=head1 NAME

    AI::MXNet::Gluon::RNN::GRUCell
=cut

=head1 DESCRIPTION

    Gated Rectified Unit (GRU) network cell.
    Note: this is an implementation of the cuDNN version of GRUs
    (slight modification compared to Cho et al. 2014).

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol.
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
    prefix : str, default 'gru_'
        prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
=cut

has 'hidden_size' => (is => 'rw', isa => 'Int', required => 1);
has [qw/
    i2h_weight_initializer
    h2h_weight_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]');
has [qw/
    i2h_bias_initializer
    h2h_bias_initializer
    /]            => (is => 'rw', isa => 'Maybe[Initializer]', default => 'zeros');
has 'input_size'  => (is => 'rw', isa => 'Int', default => 0);
has [qw/
        i2h_weight
        h2h_weight
        i2h_bias
        h2h_bias
    /]            => (is => 'rw', init_arg => undef);

method python_constructor_arguments()
{
    [qw/
        hidden_size
        i2h_weight_initializer h2h_weight_initializer
        i2h_bias_initializer h2h_bias_initializer
        input_size
    /];
}

sub BUILD
{
    my $self = shift;
    $self->i2h_weight($self->params->get(
        'i2h_weight', shape=>[3*$self->hidden_size, $self->input_size],
        init => $self->i2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_weight($self->params->get(
        'h2h_weight', shape=>[3*$self->hidden_size, $self->hidden_size],
        init => $self->h2h_weight_initializer,
        allow_deferred_init => 1
    ));
    $self->i2h_bias($self->params->get(
        'i2h_bias', shape=>[3*$self->hidden_size],
        init => $self->i2h_bias_initializer,
        allow_deferred_init => 1
    ));
    $self->h2h_bias($self->params->get(
        'h2h_bias', shape=>[3*$self->hidden_size],
        init => $self->h2h_bias_initializer,
        allow_deferred_init => 1
    ));
}

method state_info(Int $batch_size=0)
{
    return [{ shape => [$batch_size, $self->hidden_size], __layout__ => 'NC' }];
}

method _alias() { 'gru' }

method hybrid_forward(
    GluonClass $F, GluonInput $inputs, GluonInput $states,
    GluonInput :$i2h_weight, GluonInput :$h2h_weight, GluonInput :$i2h_bias, GluonInput :$h2h_bias
)
{
    my $prefix = "t${\ $self->counter}_";
    my $prev_state_h = $states->[0];
    my $i2h = $F->FullyConnected(
        $inputs, $i2h_weight, $i2h_bias,
        num_hidden => $self->hidden_size*3,
        name => "${prefix}i2h"
    );
    my $h2h = $F->FullyConnected(
        $states->[0], $h2h_weight, $h2h_bias,
        num_hidden => $self->hidden_size*3,
        name => "${prefix}h2h"
    );
    my ($i2h_r, $i2h_z, $h2h_r, $h2h_z);
    ($i2h_r, $i2h_z, $i2h) = @{ $F->SliceChannel($i2h, num_outputs => 3, name => "${prefix}i2h_slice") };
    ($h2h_r, $h2h_z, $h2h) = @{ $F->SliceChannel($h2h, num_outputs => 3, name => "${prefix}h2h_slice") };
    my $reset_gate  = $F->Activation($i2h_r + $h2h_r, act_type=>"sigmoid", name => "${prefix}r_act");
    my $update_gate = $F->Activation($i2h_z + $h2h_z, act_type=>"sigmoid", name => "${prefix}z_act");
    my $next_h_tmp = $F->Activation($i2h + $reset_gate * $h2h, act_type => "tanh", name => "${prefix}h_act");
    my $next_h = $F->_plus((1 - $update_gate) * $next_h_tmp, $update_gate * $prev_state_h, name => "${prefix}out");
    return ($next_h, [$next_h]);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::SequentialRNNCell;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
no warnings 'redefine';
extends 'AI::MXNet::Gluon::Block';
with 'AI::MXNet::Gluon::RNN::RecurrentCell';
has 'modified'      => (is => 'rw', isa => 'Bool', default => 0);
has [qw/counter
     init_counter/] => (is => 'rw', isa => 'Int', default => -1);

sub BUILD
{
    my $self = shift;
    $self->reset;
}

=head1 NAME

    AI::MXNet::Gluon::RNN::SequentialRNNCell
=cut

=head1 DESCRIPTION

    Sequentially stacking multiple RNN cells.
=cut

=head2 add

    Appends a cell into the stack.

    Parameters
    ----------
        cell : rnn cell
=cut

method add(AI::MXNet::Gluon::Block $cell)
{
    $self->register_child($cell);
}

method state_info(Int $batch_size=0)
{
    return $self->_cells_state_info($self->_children, $batch_size);
}

method begin_state(%kwargs)
{
    assert(
        (not $self->modified),
        "After applying modifier cells (e.g. ZoneoutCell) the base ".
        "cell cannot be called directly. Call the modifier cell instead."
    );
    return $self->_cells_begin_state($self->_children, %kwargs);
}

method unroll(Int $length, GluonInput $inputs, Maybe[GluonInput] :$begin_state=, Str :$layout='NTC', Maybe[Bool] :$merge_outputs=)
{
    $self->reset();
    my ($F, $batch_size);
    ($inputs, undef, $F, $batch_size) = $self->_format_sequence($length, $inputs, $layout, undef);
    my $num_cells = @{ $self->_children };
    $begin_state = $self->_get_begin_state($F, $begin_state, $inputs, $batch_size);
    my $p = 0;
    my @next_states;
    my $states;
    enumerate(sub {
        my ($i, $cell) = @_;
        my $n = @{ $cell->state_info() };
        $states = [@{ $begin_state }[$p..$p+$n-1]];
        $p += $n;
        ($inputs, $states) = $cell->unroll(
            $length, $inputs, begin_state => $states, layout => $layout,
            merge_outputs => ($i < ($num_cells - 1)) ? undef : $merge_outputs
        );
        push @next_states, @{ $states };
    }, $self->_children);
    return ($inputs, \@next_states);
}

method call($inputs, $states)
{
    $self->counter($self->counter + 1);
    my @next_states;
    my $p = 0;
    for my $cell (@{ $self->_children })
    {
        assert(not $cell->isa('AI::MXNet::Gluon::RNN::BidirectionalCell'));
        my $n = @{ $cell->state_info() };
        my $state = [@{ $states }[$p,$p+$n-1]];
        $p += $n;
        ($inputs, $state) = $cell->($inputs, $state);
        push @next_states, @{ $state };
    }
    return ($inputs, \@next_states);
}

use overload '@{}' => sub { shift->_children };
use overload '""'  => sub {
    my $self = shift;
    my $s = "%s(\n%s\n)";
    my @children;
    enumerate(sub {
        my ($i, $m) = @_;
        push @children, "($i): ". AI::MXNet::Base::_indent("$m", 2);
    }, $self->_children);
    return sprintf($s, $self->_class_name, join("\n", @children));
};

method hybrid_forward(@args)
{
    confess('Not Implemented');
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::DropoutCell;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';

=head1 NAME

    AI::MXNet::Gluon::RNN::DropoutCell
=cut

=head1 DESCRIPTION

    Applies dropout on input.

    Parameters
    ----------
    rate : float
        Percentage of elements to drop out, which
        is 1 - percentage to retain.
=cut

has 'rate' => (is => 'ro', isa => 'Num', required => 1);
method python_constructor_arguments() { ['rate'] }

method state_info(Int $batch_size=0) { [] }

method _alias() { 'dropout' }

method hybrid_forward(GluonClass $F, GluonInput $inputs, GluonInput $states)
{
    if($self->rate > 0)
    {
        $inputs = $F->Dropout($inputs, p => $self->rate, name => "t${\ $self->counter }_fwd");
    }
    return ($inputs, $states);
}

method unroll(Int $length, GluonInput $inputs, Maybe[GluonInput] :$begin_state=, Str :$layout='NTC', Maybe[Bool] :$merge_outputs=)
{
    $self->reset;
    my $F;
    ($inputs, undef, $F) = $self->_format_sequence($length, $inputs, $layout, $merge_outputs);
    if(blessed $inputs)
    {
        return $self->hybrid_forward($F, $inputs, $begin_state//[]);
    }
    else
    {
        return $self->SUPER::unroll(
            $length, $inputs, begin_state => $begin_state, layout => $layout,
            merge_outputs => $merge_outputs
        );
    }
}

use overload '""' => sub {
    my $self = shift;
    return $self->_class_name.'(rate ='.$self->rate.')';
};

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::ModifierCell;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';
has 'base_cell' => (is => 'rw', isa => 'AI::MXNet::Gluon::RNN::HybridRecurrentCell', required => 1);

=head1 NAME

    AI::MXNet::Gluon::RNN::ModifierCell
=cut

=head1 DESCRIPTION

    Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Zoneout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifier cell
    should be used instead.
=cut


sub BUILD
{
    my $self = shift;
    assert(
        (not $self->base_cell->modified),
        "Cell ${\ $self->base_cell->name } is already modified. One cell cannot be modified twice"
    );
    $self->base_cell->modified(1);
}

method params()
{
    return $self->base_cell->params;
}

method state_info(Int $batch_size=0)
{
    return $self->base_cell->state_info($batch_size);

}

method begin_state(CodeRef :$func=sub{ AI::MXNet::Symbol->zeros(@_) }, %kwargs)
{
    assert(
        (not $self->modified),
        "After applying modifier cells (e.g. DropoutCell) the base ".
        "cell cannot be called directly. Call the modifier cell instead."
    );
    $self->base_cell->modified(0);
    my $begin = $self->base_cell->begin_state(func => $func, %kwargs);
    $self->base_cell->modified(1);
    return $begin;
}

method hybrid_forward(GluonClass $F, GluonInput $inputs, GluonInput $states)
{
    confess('Not Implemented');
}

use overload '""' => sub {
    my $self = shift;
    return $self->_class_name.'('.$self->base_cell.')';
};

package AI::MXNet::Gluon::RNN::ZoneoutCell;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::RNN::ModifierCell';

=head1 NAME

    AI::MXNet::Gluon::RNN::ZoneoutCell
=cut

=head1 DESCRIPTION

    Applies Zoneout on base cell.
=cut
has [qw/zoneout_outputs
        zoneout_states/] => (is => 'ro', isa => 'Num', default => 0);
has 'prev_output' => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { ['base_cell', 'zoneout_outputs', 'zoneout_states'] }

sub BUILD
{
    my $self = shift;
    assert(
        (not $self->base_cell->isa('AI::MXNet::Gluon::RNN::BidirectionalCell')),
        "BidirectionalCell doesn't support zoneout since it doesn't support step. ".
        "Please add ZoneoutCell to the cells underneath instead."
    );
    assert(
        (not $self->base_cell->isa('AI::MXNet::Gluon::RNN::SequentialRNNCel') or not $self->base_cell->bidirectional),
        "Bidirectional SequentialRNNCell doesn't support zoneout. ".
        "Please add ZoneoutCell to the cells underneath instead."
    );
}

use overload '""' => sub {
    my $self = shift;
    return $self->_class_name.'(p_out='.$self->zoneout_outputs.', p_state='.$self->zoneout_states.
           ', '.$self->base_cell.')';
};

method _alias() { 'zoneout' }

method reset()
{
    $self->SUPER::reset();
    $self->prev_output(undef);
}

method hybrid_forward(GluonClass $F, GluonInput $inputs, GluonInput $states)
{
    my ($cell, $p_outputs, $p_states) = ($self->base_cell, $self->zoneout_outputs, $self->zoneout_states);
    my ($next_output, $next_states) = $cell->($inputs, $states);
    my $mask = sub { my ($p, $like) = @_; $F->Dropout($F->ones_like($like), p=>$p) };

    my $prev_output = $self->prev_output//$F->zeros_like($next_output);
    my $output = $p_outputs != 0 ? $F->where($mask->($p_outputs, $next_output), $next_output, $prev_output) : $next_output;
    if($p_states != 0)
    {
        my @tmp;
        for(zip($next_states, $states)) {
            my ($new_s, $old_s) = @$_;
            push @tmp, $F->where($mask->($p_states, $new_s), $new_s, $old_s);
        }
        $states = \@tmp;
    }
    else
    {
        $states = $next_states;
    }
    $self->prev_output($output);
    return ($output, $states);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::ResidualCell;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::RNN::ModifierCell';
method python_constructor_arguments() { ['base_cell'] }

=head1 NAME

    AI::MXNet::Gluon::RNN::ResidualCell
=cut

=head1 DESCRIPTION

    Adds residual connection as described in Wu et al, 2016
    (https://arxiv.org/abs/1609.08144).
    Output of the cell is output of the base cell plus input.
=cut

method hybrid_forward(GluonClas $F, GluonInput $inputs, GluonInput $states)
{
    my $output;
    ($output, $states) = $self->base_cell->($inputs, $states);
    $output = $F->elemwise_add($output, $inputs, name => "t${\ $self->counter }_fwd");
    return ($output, $states);
}

method unroll(Int $length, GluonInput $inputs, Maybe[GluonInput] :$begin_state=, Str :$layout='NTC', Maybe[Bool] :$merge_outputs=)
{
    $self->reset();

    $self->base_cell->modified(0);
    my ($outputs, $states) = $self->base_cell->unroll(
        $length, $inputs, begin_state => $begin_state, layout => $layout, merge_outputs => $merge_outputs
    );
    $self->base_cell->modified(1);

    $merge_outputs //= blessed $outputs ? 1 : 0;
    my $F;
    ($inputs, undef, $F) = $self->_format_sequence($length, $inputs, $layout, $merge_outputs);
    if($merge_outputs)
    {
        $outputs = $F->elemwise_add($outputs, $inputs);
    }
    else
    {
        my @tmp;
        for(zip($outputs, $inputs)) {
            my ($i, $j) = @$_;
            push @tmp, $F->elemwise_add($i, $j);
        }
        $outputs = \@tmp;
    }
    return ($outputs, $states);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

package AI::MXNet::Gluon::RNN::BidirectionalCell;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';
has [qw/l_cell r_cell/] => (is => 'ro', isa => 'AI::MXNet::Gluon::RNN::HybridRecurrentCell', required => 1);
has 'output_prefix'     => (is => 'ro', isa => 'Str', default => 'bi_');
method python_constructor_arguments() { ['l_cell', 'r_cell', 'output_prefix'] }

=head1 NAME

    AI::MXNet::Gluon::RNN::BidirectionalCell
=cut

=head1 DESCRIPTION

    Bidirectional RNN cell.

    Parameters
    ----------
    l_cell : RecurrentCell
        Cell for forward unrolling
    r_cell : RecurrentCell
        Cell for backward unrolling
=cut

method call($inputs, $states)
{
    confess("Bidirectional cell cannot be stepped. Please use unroll");
}

use overload '""' => sub {
    my $self = shift;
    "${\ $self->_class_name }(forward=${\ $self->l_cell }, backward=${\ $self->r_cell })";
};

method state_info(Int $batch_size=0)
{
    return $self->_cells_state_info($self->_children, $batch_size);
}

method begin_state(%kwargs)
{
    assert(
        (not $self->modified),
        "After applying modifier cells (e.g. DropoutCell) the base ".
        "cell cannot be called directly. Call the modifier cell instead."
    );
    return $self->_cells_begin_state($self->_children, %kwargs);
}

method unroll(Int $length, GluonInput $inputs, Maybe[GluonInput] :$begin_state=, Str :$layout='NTC', Maybe[Bool] :$merge_outputs=)
{
    $self->reset();
    my ($axis, $F, $batch_size);
    ($inputs, $axis, $F, $batch_size) = $self->_format_sequence($length, $inputs, $layout, 0);
    $begin_state //= $self->_get_begin_state($F, $begin_state, $inputs, $batch_size);

    my $states = $begin_state;
    my ($l_cell, $r_cell) = @{ $self->_children };
    $l_cell->state_info($batch_size);
    my ($l_outputs, $l_states) = $l_cell->unroll(
            $length, $inputs,
            begin_state => [@{ $states }[0..@{ $l_cell->state_info($batch_size) }-1]],
            layout => $layout,
            merge_outputs => $merge_outputs
    );
    my ($r_outputs, $r_states) = $r_cell->unroll(
        $length, [reverse @{$inputs}],
        begin_state     => [@{$states}[@{ $l_cell->state_info }..@{$states}-1]],
        layout          => $layout,
        merge_outputs   => $merge_outputs
    );
    if(not defined $merge_outputs)
    {
        $merge_outputs = blessed $l_outputs and blessed $r_outputs;
        ($l_outputs) = $self->_format_sequence(undef, $l_outputs, $layout, $merge_outputs);
        ($r_outputs) = $self->_format_sequence(undef, $r_outputs, $layout, $merge_outputs);
    }
    my $outputs;
    if($merge_outputs)
    {
        $r_outputs = $F->reverse($r_outputs, axis=>$axis);
        $outputs = $F->concat($l_outputs, $r_outputs, dim=>2, name=>$self->output_prefix.'out');
    }
    else
    {
        $outputs = [];
        enumerate(sub {
            my ($i, $l_o, $r_o) = @_;
                push @$outputs, $F->concat(
                    $l_o, $r_o, dim=>1,
                    name => sprintf('%st%d', $self->output_prefix, $i)
                );
            }, [@{ $l_outputs }], [reverse(@{ $r_outputs })]
        );
    }
    $states = [@{ $l_states }, @{ $r_states }];
    return ($outputs, $states);
}

__PACKAGE__->register('AI::MXNet::Gluon::RNN');

1;
