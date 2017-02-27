package AI::MXNet::RNN::Params;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::RNN::Params
=cut

=head1 DESCRIPTION

    Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        All variables' name created by this container will
        be prepended with prefix
=cut
has '_prefix' => (is => 'ro', init_arg => 'prefix', isa => 'Str', default => '');
has '_params' => (is => 'rw', init_arg => undef);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(prefix => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    $self->_params({});
}


=head2 get

        Get a variable with name or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        @kwargs:
            more arguments that are passed to mx->sym->Variable call
=cut

method get(Str $name, @kwargs)
{
    $name = $self->_prefix . $name;
    if(not exists $self->_params->{$name})
    {
        $self->_params->{$name} = AI::MXNet::Symbol->Variable($name, @kwargs);
    }
    return $self->_params->{$name};
}

package AI::MXNet::RNN::Cell::Base;
=head1 NAME

    AI::MXNet::RNNCell::Base
=cut

=head1 DESCRIPTION

    Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str
        prefix for name of layers
        (and name of weight if params is undef)
    params : RNNParams or undef
        container for weight sharing between cells.
        created if undef.
=cut

use AI::MXNet::Base;
use Mouse;
use overload "&{}"  => sub { my $self = shift; sub { $self->call(@_) } };
has '_prefix'       => (is => 'rw', init_arg => 'prefix', isa => 'Str', default => '');
has '_params'       => (is => 'rw', init_arg => 'params', isa => 'Maybe[AI::MXNet::RNN::Params]');
has [qw/_own_params
        _modified
        _init_counter
        _counter
                 /] => (is => 'rw', init_arg => undef);

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(prefix => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    if(not defined $self->_params)
    {
        $self->_own_params(1);
        $self->_params(AI::MXNet::RNN::Params->new($self->_prefix));
    }
    else
    {
        $self->_own_params(0);
    }
    $self->_modified(0);
    $self->reset;
}

=head2 reset

    Reset before re-using the cell for another graph
=cut

method reset()
{
    $self->_init_counter(-1);
    $self->_counter(-1);
}

=head2 call

        Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : mx->sym->Variable
            input symbol, 2D, batch * num_units
        states : mx->sym->Variable or ArrayRef[Symbol]
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        Can be called via overloaded &{}: &{$cell}($inputs, $states);
=cut

method call(AI::MXNet::Symbol $inputs, AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol] $states)
{
    confess("Not Implemented");
}

=head2 params

        Parameters of this cell
=cut

method params()
{
    $self->_own_params(0);
    return $self->_params;
}

=head2 state_shape

        shape(s) of states
=cut

method state_shape()
{
    confess("Not Implemented");
}

=head2 begin_state

        Initial state for this cell.

        Parameters
        ----------
        func : sub ref, default AI::MXNet::Symbol->can('zeros')
            Function for creating initial state.
            Can be AI::MXNet::Symbol->can('zeros'),
            AI::MXNet::Symbol->can('uniform'), AI::MXNet::Symbol->can('Variable') etc.
            Use AI::MXNet::Symbol->can('Variable') if you want to directly
            feed input as states.
        @kwargs :
            more keyword arguments passed to func. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : array ref of Symbol
            starting states for first RNN step
=cut

method begin_state(CodeRef $func=AI::MXNet::Symbol->can('zeros'), @kwargs)
{
    assert(
        (not $self->_modified),
        "After applying modifier cells (e.g. DropoutCell) the base "
        ."cell cannot be called directly. Call the modifier cell instead."
    );
    my @states;
    my $func_needs_named_name = $func ne AI::MXNet::Symbol->can('Variable');
    for my $shape (@{ $self->state_shape })
    {
        $self->_init_counter($self->_init_counter + 1);
        my @name = (sprintf("%sbegin_state_%d", $self->_prefix, $self->_init_counter));
        if($func_needs_named_name)
        {
            unshift(@name, 'name');
        }
        my $state = &{$func}(
            'AI::MXNet::Symbol',
            @name,
            (defined $shape ? (shape => $shape) : ()),
            @kwargs
        );
        push @states, $state;
    }
    return \@states;
}

=head2 unpack_weights

        Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : hash ref of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : hash ref of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
=cut

method unpack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    return { %{ $args } };
}

=head2 pack_weights

        Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : hash ref of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : hash ref of str -> NDArray
            dictionary with weights associated to
            this cell packed.
=cut

method pack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    return { %{ $args } };
}

=head2 unroll

        Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or undef
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If inputs is a array ref of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).

            If inputs is undef, Placeholder variables are
            automatically created.
        begin_state : array ref of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if undef.
        input_prefix : str
            prefix for automatically created input
            placehodlers.
        layout : str
            layout of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool
            if 0, return outputs as a list of Symbols.
            If 1, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

        Returns
        -------
        outputs : array ref of Symbol or Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
=cut


method unroll(
    Int $length,
    Maybe[AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]] :$inputs=,
    Maybe[AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]] :$begin_state=,
    Str                                                  :$input_prefix='',
    Str                                                  :$layout='NTC',
    Bool                                                 :$merge_outputs=0
)
{
    $self->reset;
    my $axis = index($layout, 'T');
    if(not defined $inputs)
    {
        $inputs = [
            map { AI::MXNet::Symbol->Variable("${input_prefix}t${_}_data") } (0..$length-1)
        ];
    }
    elsif(blessed($inputs))
    {
        assert(
            (@{ $inputs->list_outputs() } == 1),
            "unroll doesn't allow grouped symbol as input. Please "
            ."convert to list first or let unroll handle slicing"
        );
        $inputs = AI::MXNet::Symbol->SliceChannel(
            $inputs,
            axis         => $axis,
            num_outputs  => $length,
            squeeze_axis => 1
        );
    }
    else
    {
        assert(@$inputs == $length);
    }
    $begin_state //= $self->begin_state;
    my $states = $begin_state;
    my $outputs;
    my @inputs = @{ $inputs };
    for my $i (0..$length-1)
    {
        my ($output, $states) = &{$self}(
            $inputs[$i],
            $states
        );
        push @$outputs, $output;
    }
    if($merge_outputs)
    {
        @$outputs = map { AI::MXNet::Symbol->expand_dims($_, axis => $axis) } @$outputs;
        $outputs = AI::MXNet::Symbol->Concat(@$outputs, dim => $axis);
    }
    return($outputs, $states);
}

method _get_activation($inputs, $activation, @kwargs)
{
    if(not ref $activation)
    {
        return AI::MXNet::Symbol->Activation($inputs, act_type => $activation, @kwargs);
    }
    else
    {
        return &{$activation}($inputs, @kwargs);
    }
}

package AI::MXNet::RNN::Cell;
use Mouse;
extends 'AI::MXNet::RNN::Cell::Base';

=head1 NAME 

    AI::MXNet::RNN::Cell
=cut

=head1 DESCRIPTION

    Simple recurrent neural network cell

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    activation : str or Symbol, default 'tanh'
        type of activation function
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is undef)
    params : AI::MXNet::RNNParams or undef
        container for weight sharing between cells.
        created if undef.
=cut

has '_num_hidden'   => (is => 'ro', init_arg => 'num_hidden', isa => 'Int', required => 1);
has '_activation'  => (
    is       => 'ro',
    init_arg => 'activation',
    isa      => 'Activation',
    default  => 'tanh'
);
has '+_prefix'    => (default => 'rnn_');
has [qw/_iW _iB
        _hW _hB/] => (is => 'rw', init_arg => undef);

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(num_hidden => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    $self->_iW($self->params->get('i2h_weight'));
    $self->_iB($self->params->get('i2h_bias'));
    $self->_hW($self->params->get('h2h_weight'));
    $self->_hB($self->params->get('h2h_bias'));
}

=head2 state_shape

        shape(s) of states
=cut

method state_shape()
{
    return [[0, $self->_num_hidden]];
}

=head2 call

        Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
=cut

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    $self->_counter($self->_counter + 1);
    my $name = sprintf('%st%d_', $self->_prefix, $self->_counter);
    my $i2h = AI::MXNet::Symbol->FullyConnected(
        data       => $inputs,
        weight     => $self->_iW,
        bias       => $self->_iB,
        num_hidden => $self->_num_hidden,
        name       => "${name}i2h"
    );
    my $h2h = AI::MXNet::Symbol->FullyConnected(
        data       => @{$states}[0],
        weight     => $self->_hW,
        bias       => $self->_hB,
        num_hidden => $self->_num_hidden,
        name       => "${name}h2h"
    );
    my $output = $self->_get_activation(
        $i2h + $h2h,
        $self->_activation,
        name       => "${name}out"
    );
    return ($output, [$output]);
}

package AI::MXNet::RNN::LSTMCell;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::RNN::Cell';

=head1 NAME 

    AI::MXNet::RNN::LSTMCell
=cut

=head1 DESCRIPTION

    Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'lstm_'
        prefix for name of layers
        (and name of weight if params is undef)
    params : AI::MXNet::RNN::Params or None
        container for weight sharing between cells.
        created if undef.
=cut

has '+_prefix'     => (default => 'lstm_');
has '+_activation' => (init_arg => undef);

=head2 state_shape

    shape(s) of states
=cut

method state_shape()
{
    return [[0, $self->_num_hidden], [0, $self->_num_hidden]];
}

=head2 unpack_weights

        Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : hashref of str -> NDArray
            dictionary containing packed weights.
            usually from $Module->get_output()

        Returns
        -------
        args : hashref of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
=cut

method unpack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    $args = { %{ $args } };
    my $outs = ['_i', '_f', '_c', '_o'];
    my $h = $self->_num_hidden;
    for my $i ('i2h', 'h2h')
    {
        my $weight = delete $args->{ sprintf('%s%s_weight', $self->_prefix, $i) };
        my $bias   = delete $args->{ sprintf('%s%s_bias', $self->_prefix, $i) };
        enumerate(sub {
            my ($j, $name) = @_;
            my $wname = sprintf('%s%s%s_weight', $self->_prefix, $i, $name);
            $args->{$wname} = $weight->slice([$j*$h,($j+1)*$h-1])->copy;
            my $bname = sprintf('%s%s%s_bias', $self->_prefix, $i, $name);
            $args->{$bname} = $bias->slice([$j*$h,($j+1)*$h-1])->copy;
        }, $outs);
    }
    return $args;
}

=head2 pack_weights

        Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : hashref of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : hashref of str -> NDArray
            dictionary with weights associated to
            this cell packed.
=cut


method pack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    $args = { %{ $args } };
    my @outs = ('_i', '_f', '_c', '_o');
    my $h = $self->_num_hidden;
    for my $i ('i2h', 'h2h')
    {
        my @weight;
        my @bias;
        for my $name (@outs)
        {
            my $wname = sprintf('%s%s%s_weight', $self->_prefix, $i, $name);
            push @weight, delete $args->{$wname};
            my $bname = sprintf('%s%s%s_bias', $self->_prefix, $i, $name);
            push @bias, delete $args->{$bname};
        }
        $args->{ sprintf('%s%s_weight', $self->_prefix, $i) } = AI::MXNet::NDArray->concatenate(
            \@weight
        );
        $args->{ sprintf('%s%s_bias', $self->_prefix, $i) } = AI::MXNet::NDArray->concatenate(
            \@bias
        );
    }
    return $args;
}

=head2 call

        Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
=cut

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    $self->_counter($self->_counter + 1);
    my $name = sprintf('%st%d_', $self->_prefix, $self->_counter);
    my @states = @{ $states };
    my $i2h = AI::MXNet::Symbol->FullyConnected(
        data       => $inputs,
        weight     => $self->_iW,
        bias       => $self->_iB,
        num_hidden => $self->_num_hidden*4,
        name       => "${name}i2h"
    );
    my $h2h = AI::MXNet::Symbol->FullyConnected(
        data       => $states[0],
        weight     => $self->_hW,
        bias       => $self->_hB,
        num_hidden => $self->_num_hidden*4,
        name       => "${name}h2h"
    );
    my $gates = $i2h + $h2h;
    my @slice_gates = @{ AI::MXNet::Symbol->SliceChannel(
        $gates, num_outputs => 4, name => "${name}slice"
    ) };
    my $in_gate = AI::MXNet::Symbol->Activation(
        $slice_gates[0], act_type => "sigmoid", name => "${name}i"
    );
    my $forget_gate = AI::MXNet::Symbol->Activation(
        $slice_gates[1], act_type => "sigmoid", name => "${name}f"
    );
    my $in_transform = AI::MXNet::Symbol->Activation(
        $slice_gates[2], act_type => "tanh", name => "${name}c"
    );
    my $out_gate = AI::MXNet::Symbol->Activation(
        $slice_gates[3], act_type => "sigmoid", name => "${name}o"
    );
    my $next_c = AI::MXNet::Symbol->_plus(
        $forget_gate * $states[1], $in_gate * $in_transform,
        name => "{$name}state"
    );
    my $next_h = AI::MXNet::Symbol->_mul(
        $out_gate,
        AI::MXNet::Symbol->Activation(
            $next_c, act_type => "tanh"
        ),
        name => "${name}out"
    );
    return ($next_h, [$next_h, $next_c]);

}

package AI::MXNet::RNN::FusedCell;
use Mouse;
use AI::MXNet::Types;
use AI::MXNet::Base;
extends 'AI::MXNet::RNN::Cell::Base';

=head1 NAME

    AI::MXNet::RNN::FusedCell
=cut

=head1 DESCRIPTION

    Fusing RNN layers across time step into one kernel.
    Improves speed but is less flexible. Currently only
    supported if using cuDNN on GPU.
=cut

has '_num_hidden'      => (is => 'ro', isa => 'Int',  init_arg => 'num_hidden',     required => 1);
has '_num_layers'      => (is => 'ro', isa => 'Int',  init_arg => 'num_layers',     default => 1);
has '_dropout'         => (is => 'ro', isa => 'Num',  init_arg => 'dropout',        default => 0);
has '_get_next_state'  => (is => 'ro', isa => 'Bool', init_arg => 'get_next_state', default => 0);
has '_bidirectional'   => (is => 'ro', isa => 'Bool', init_arg => 'bidirectional',  default => 0);
has 'initializer'      => (is => 'rw', isa => 'Maybe[AI::MXNet::Initializer]');
has '_mode'            => (
    is => 'ro',
    isa => enum([qw/rnn_relu rnn_tanh lstm gru/]),
    init_arg => 'mode',
    default => 'lstm'
);
has [qw/_parameter
        _directions
        _weight_names
        _num_weights/] => (is => 'rw', init_arg => undef);

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(num_hidden => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    if(not $self->_prefix)
    {
        $self->_prefix($self->_mode.'_');
    }
    if(not defined $self->initializer)
    {
        $self->initializer(
            AI::MXNet::Xavier->new(
                factor_type => 'in',
                magnitude   => 2.34
            )
        );
    }
    if(not $self->initializer->isa('AI::MXNet::FusedRNN'))
    {
        $self->initializer(
            AI::MXNet::FusedRNN->new(
                init           => $self->initializer,
                num_hidden     => $self->_num_hidden,
                num_layers     => $self->_num_layers,
                mode           => $self->_mode,
                bidirectional  => $self->_bidirectional
            )
        );
    }
    $self->_parameter($self->params->get('parameters', init => $self->initializer));
    $self->_directions($self->_bidirectional ? 2 : 1);
    $self->_weight_names({
        rnn_relu => [''],
        rnn_tanh => [''],
        lstm     => ['_i', '_f', '_c', '_o'],
        gru      => ['_r', '_z', '_o']
    }->{ $self->_mode });
    $self->_num_weights(scalar@{ $self->_weight_names });
}


method state_shape()
{
    my $b = $self->_directions;
    my $n = $self->_mode eq 'lstm' ? 2 : 1;
    return [([$b*$self->_num_layers, 0, $self->_num_hidden])x$n];
}

# slice fused rnn weights
method _slice_weights($arr, $li, $lh)
{
    my %args;
    my $b = $self->_directions;
    my $m = $self->_num_weights;
    my @c = @{ $self->_weight_names };
    my @d = ('l', 'r');

    my $p = 0;
    for my $i (0..$self->_num_layers-1)
    {
        for my $j (0..$b-1)
        {
            for my $k (0..$m-1)
            {
                my $name = sprintf('%s%s%d_i2h%s_weight', $self->_prefix, $d[$j], $i, $c[$k]);
                my $size;
                if($i > 0)
                {
                    $size = $b*$lh*$lh;
                    $args{$name} = $arr->slice([$p,$p+$size-1])->reshape([$lh, $b*$lh]);
                }
                else
                {
                    $size = $li*$lh;
                    $args{$name} = $arr->slice([$p,$p+$size-1])->reshape([$lh, $li]);
                }
                $p += $size;
            }
            for my $k (0..$m-1)
            {
                my $name = sprintf('%s%s%d_h2h%s_weight', $self->_prefix, $d[$j], $i, $c[$k]);
                my $size = $lh**2;
                $args{$name} = $arr->slice([$p,$p+$size-1])->reshape([$lh, $lh]);
                $p += $size;
            }
        }
    }
    for my $i (0..$self->_num_layers-1)
    {
        for my $j (0..$b-1)
        {
            for my $k (0..$m-1)
            {
                my $name = sprintf('%s%s%d_i2h%s_bias', $self->_prefix, $d[$j], $i, $c[$k]);
                $args{$name} = $arr->slice([$p,$p+$lh-1]);
                $p += $lh;
            }
            for my $k (0..$m-1)
            {
                my $name = sprintf('%s%s%d_h2h%s_bias', $self->_prefix, $d[$j], $i, $c[$k]);
                $args{$name} = $arr->slice([$p,$p+$lh-1]);
                $p += $lh;
            }
        }
    }
    assert($p == $arr->size, "Invalid parameters size for FusedRNNCell");
    return %args;
}

method unpack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    my %args = %{ $args };
    my $arr = delete $args->{ $self->_parameter->name };
    my $b = $self->_directions;
    my $m = $self->_num_weights;
    my $h = $self->_num_hidden;
    my $num_input = int(int(int($arr->size/$b)/$h)/$m) - ($self->_num_layers - 1)*($h+$b*$h+2) - $h - 2;
    my %nargs = $self->_slice_weights($arr, $num_input, $self->_num_hidden);
    %args = (%args, map { $_ => $nargs{$_}->copy } keys %nargs);
    return \%args
}

method pack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    my %args = %{ $args };
    my $b = $self->_directions;
    my $m = $self->_num_weights;
    my @c = @{ $self->_weight_names };
    my $h = $self->_num_hidden;
    my $w0 = $args{ sprintf('%sl0_i2h%s_weight', $self->_prefix, $c[0]) };
    my $num_input = $w0->shape->[1];
    my $total = ($num_input+$h+2)*$h*$m*$b + ($self->_num_layers-1)*$m*$h*($h+$b*$h+2)*$b;
    my $arr = AI::MXNet::NDArray->zeros([$total], ctx => $w0->context, dtype => $w0->dtype);
    my %nargs = $self->_slice_weights($arr, $num_input, $h);
    while(my ($name, $nd) = each %nargs)
    {
        $nd .= delete $args{ $name };
    }
    $args{ $self->_parameter->name } = $arr;
    return \%args;
}

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    confess("AI::MXNet::RNN::FusedCell cannot be stepped. Please use unroll");
}

=head2 unroll

        Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, array ref of Symbol, or undef
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            using 'TNC' is more efficient for RNN::FusedCell.

            If inputs is a array ref of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...). using single symbol is
            more efficient for RNN::FusedCell.

            If inputs is undef, a single placeholder variable is
            automatically created.
        begin_state : array ref of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if undef.
        input_prefix : str
            prefix for automatically created input
            placehodlers.
        layout : str
            layout of input/output symbol.
        merge_outputs : Bool
            default 0

        Returns
        -------
        outputs : array ref of Symbol
            output symbols.
        states : Symbol or array ref of Symbol
            has the same structure as begin_state()
=cut

method unroll(
    Int $length,
    Maybe[AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]] :$inputs=,
    Maybe[AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]] :$begin_state=,
    Str                                                  :$input_prefix='',
    Str                                                  :$layout='NTC',
    Bool                                                 :$merge_outputs=0
)
{
    $self->reset;
    my $axis = index($layout, 'T');
    $inputs //= AI::MXNet::Symbol->Variable("${input_prefix}data");
    if(blessed($inputs))
    {
        assert(
            (@{ $inputs->list_outputs() } == 1),
            "unroll doesn't allow grouped symbol as input. Please "
            ."convert to list first or let unroll handle slicing"
        );
        if($axis == 1)
        {
            AI::MXNet::Logging->warning(
                "NTC layout detected. Consider using "
                ."TNC for RNN::FusedCell for faster speed"
            );
            $inputs = AI::MXNet::Symbol->SwapAxis($inputs, dim1 => 0, dim2 => 1);
        }
        else
        {
            assert($axis == 0, "Unsupported layout $layout");
        }
    }
    else
    {
        assert(@$inputs == $length);
        $inputs = [map { AI::MXNet::Symbol->expand_dims($_, axis => 0) } @{ $inputs }];
        $inputs = AI::MXNet::Symbol->Concat(@{ $inputs }, dim => 0);
    }
    $begin_state //= $self->begin_state;
    my $states = $begin_state;
    my @states = @{ $states };
    my %states;
    if($self->_mode eq 'lstm')
    {
        %states = (state => $states[0], state_cell => $states[1]);
    }
    else
    {
        %states = (state => $states[0]);
    }
    my $rnn = AI::MXNet::Symbol->RNN(
        data          => $inputs,
        parameters    => $self->_parameter,
        state_size    => $self->_num_hidden,
        num_layers    => $self->_num_layers,
        bidirectional => $self->_bidirectional,
        p             => $self->_dropout,
        state_outputs => $self->_get_next_state,
        mode          => $self->_mode,
        name          => $self->_prefix.'rnn',
        %states
    );

    my $outputs;
    if(not $self->_get_next_state)
    {
        ($outputs, $states) = ($rnn, []);
    }
    elsif($self->_mode eq 'lstm')
    {
        my @rnn = @{ $rnn };
        ($outputs, $states) = ($rnn[0], [$rnn[1], $rnn[2]]);
    }
    else
    {
        my @rnn = @{ $rnn };
        ($outputs, $states) = ($rnn[0], [$rnn[1]]);
    }
    if(not $merge_outputs)
    {
        AI::MXNet::Logging->warning(
            "Call RNN::FusedCell->unroll with merge_outputs=1 "
            ."for faster speed"
        );
        $outputs = [@ {
            AI::MXNet::Symbol->SliceChannel(
                $outputs,
                axis         => 0,
                num_outputs  => $length,
                squeeze_axis => 1
            )
        }];
    }
    elsif($axis == 1)
    {
        $outputs = AI::MXNet::Symbol->SwapAxis($outputs, dim1 => 0, dim2 => 1);
    }
    return ($outputs, $states);
}

package AI::MXNet::RNN::SequentialCell;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::RNN::Cell::Base';

=head1 NAME

    AI:MXNet::RNN::SequentialCell
=cut

=head1 DESCRIPTION

    Sequentially stacking multiple RNN cells

    Parameters
    ----------
    params : RNN::Params or undef
        container for weight sharing between cells.
        created if undef.

=cut

has [qw/_override_cell_params _cells/] => (is => 'rw', init_arg => undef);


sub BUILD
{
    my ($self, $original_arguments) = @_;
    $self->_override_cell_params(defined $original_arguments->{params});
    $self->_cells([]);
}

=head2 add

        Append a cell into the stack.

        Parameters
        ----------
        cell : rnn cell
=cut

method add(AI::MXNet::RNN::Cell::Base $cell)
{
    push @{ $self->_cells }, $cell;
    if($self->_override_cell_params)
    {
        assert(
            $cell->_own_params,
            "Either specify params for SequentialRNNCell "
            ."or child cells, not both."
        );
        %{ $cell->params->_params } = (%{ $cell->params->_params }, %{ $self->params->_params });
    }
    %{ $self->params->_params } = (%{ $self->params->_params }, %{ $cell->params->_params });
}

method state_shape()
{
    return [map { @{ $_->state_shape } } $self->_cells];
}


method begin_state(@kwargs)
{
    assert(
        (not $self->_modified),
        "After applying modifier cells (e.g. DropoutCell) the base "
        ."cell cannot be called directly. Call the modifier cell instead."
    );
    return [map { @{ $_->begin_state(@kwargs) } } @{ $self->_cells }];
}

method unpack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    $args = $_->unpack_weights($args) for @{ $self->_cells };
    return $args;
}

method pack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    $args = $_->pack_weights($args) for @{ $self->_cells };
    return $args;
}

method call($inputs, $states)
{
    $self->_counter($self->_counter + 1);
    my @next_states;
    my $p = 0;
    for my $cell (@{ $self->_cells })
    {
        my $n = scalar(@{ $cell->state_shape });
        my $state = [@{ $states }[$p..$p+$n-1]];
        $p += $n;
        ($inputs, $state) = &{$cell}($inputs, $state);
        push @next_states, $state;
    }
    return ($inputs, [map { @$_} @next_states]);
}

package AI::MXNet::RNN::ModifierCell;
use Mouse;
extends 'AI::MXNet::RNN::Cell::Base';

=head1 NAME

    AI::MXNet::RNN::ModifierCell
=cut

=head1 DESCRIPTION

    Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Dropout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifer cell
    should be used instead.
=cut

has 'base_cell' => (is => 'ro', isa => 'AI::MXNet::RNN::Cell::Base', required => 1);

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(base_cell => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    $self->base_cell->_modified(1);
}

method params()
{
    $self->_own_params(0);
    return $self->base_cell->params;
}

method state_shape()
{
    return $self->base_cell->state_shape;
}

method begin_state(CodeRef $init_sym=AI::MXNet::Symbol->can('zeros'), @kwargs)
{
    assert(
        not $self->_modified,
        "After applying modifier cells (e.g. DropoutCell) the base "
        ."cell cannot be called directly. Call the modifier cell instead."
    );
    $self->base_cell->_modified(0);
    my $begin_state = $self->base_cell->begin_state($init_sym, @kwargs);
    $self->base_cell->_modified(1);
    return $begin_state;
}

method unpack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    return $self->base_cell->unpack_weights($args)
}

method pack_weights(HashRef[AI::MXNet::NDArray] $args)
{
    return $self->base_cell->pack_weights($args)
}

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    confess("Not Implemented");
}

package AI::MXNet::RNN::DropoutCell;
use Mouse;
extends 'AI::MXNet::RNN::ModifierCell';
has [qw/dropout_outputs dropout_states/] => (is => 'ro', isa => 'Num', default => 0);

=head1 NAME

    AI::MXNet::RNN::DropoutCell
=cut

=head1 DESCRIPTION

    Apply dropout on base cell
=cut

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    my ($output, $states) = &{$self->base_cell}($inputs, $states);
    if($self->dropout_outputs > 0)
    {
        $output = AI::MXNet::Symbol->Dropout(data => $output, p => $self->dropout_outputs);
    }
    if($self->dropout_states > 0)
    {
        $states = [map { AI::MXNet::Symbol->Dropout(data => $_, p => $self->dropout_states) } @{ $states }];
    }
    return ($output, $states);
}

package AI::MXNet::RNN::ZoneoutCell;
use Mouse;
extends 'AI::MXNet::RNN::ModifierCell';
has [qw/zoneout_outputs zoneout_states/] => (is => 'ro', isa => 'Num', default => 0);
has 'prev_output' => (is => 'rw', init_arg => undef);

=head1 NAME

    AI::MXNet::RNN::ZoneoutCell
=cut

=head1 DESCRIPTION

    Apply Zoneout on base cell
=cut

method call(AI::MXNet::Symbol $inputs, SymbolOrArrayOfSymbols $states)
{
    confess("Not Implemented")
}

1;