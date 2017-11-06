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


# Scope for collecting child 'Block's
use strict;
use warnings;
use AI::MXNet::Gluon::Parameter;
package AI::MXNet::Gluon::BlockScope;
use AI::MXNet::Function::Parameters;
my $_current;
use Mouse;
has '_block'      => (is => 'ro', init_arg => 'block');
has [qw/_counter _old_scope
    _name_scope/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->_counter({});
}

# Creates prefix and params for new Block.
method create($prefix, $params, $hint)
{
    my $current = $_current;
    if(not defined $current)
    {
        if(not defined $prefix)
        {
            $prefix = AI::MXNet::Symbol::NameManager->current->get(undef, $hint) . '_';
        }
        if(not defined $params)
        {
            $params = AI::MXNet::Gluon::ParameterDict->new(prefix => $prefix);
        }
        else
        {
            $params = AI::MXNet::Gluon::ParameterDict->new(prefix => $params->prefix, shared => $params);
        }
        return ($prefix, $params);
    }

    if(not defined $prefix)
    {
        my $count = $current->_counter->{ $hint } // 0;
        $prefix = sprintf('%s%d_', $hint, $count);
        $current->_counter->{$hint} = $count + 1;
    }
    if(not defined $params)
    {
        my $parent = $current->_block->params;
        $params = AI::MXNet::Gluon::ParameterDict->new(prefix => $parent->prefix.$prefix, shared => $parent->_shared);
    }
    else
    {
        $params = AI::MXNet::Gluon::ParameterDict->new(prefix => $params->prefix, $params);
    }
    return ($current->_block->prefix.$prefix, $params);
}

method __enter__()
{
    $self->_old_scope($_current);
    $_current = $self;
    $self->_name_scope(AI::MXNet::Symbol::NameManager->current);
    AI::MXNet::Symbol::NameManager->set_current(AI::MXNet::Symbol::Prefix->new(prefix => $self->_block->prefix));
    return $self;
}

method __exit__()
{
    AI::MXNet::Symbol::NameManager->set_current($self->_name_scope);
    $self->_name_scope(undef);
    $_current = $self->_old_scope;
}

package AI::MXNet::Gluon::Block;
use AI::MXNet::Gluon::Mouse;

=head2 NAME

    AI::MXNet::Gluon::Block - Base class for all neural network layers and models.

=head2 DESCRIPTION

    Base class for all neural network layers and models. Your models should
    subclass this class.

    `Block` can be nested recursively in a tree structure. You can create and
    assign child `Block` as regular attributes::

        from mxnet.gluon import Block, nn
        from mxnet import ndarray as F

        class Model(Block):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                # use name_scope to give child Blocks appropriate names.
                # It also allows sharing Parameters between Blocks recursively.
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)

                x = F.relu(self.dense0(x))
                return F.relu(self.dense1(x))

        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model(F.zeros((10, 10), ctx=mx.cpu(0)))


    Child `Block` assigned this way will be registered and `collect_params`
    will collect their Parameters recursively.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. It will be prepended to the names of all
        Parameters and child `Block`s in this `Block`'s `name_scope`. Prefix
        should be unique within one model to prevent name collisions.
    params : ParameterDict or None
        `ParameterDict` for sharing weights with the new `Block`. For example,
        if you want `dense1` to share `dense0`'s weights, you can do::

            dense0 = nn.Dense(20)
            dense1 = nn.Dense(20, params=dense0.collect_params())
=cut

method _flatten(
    $args
)
{
    if(blessed $args and $args->isa('AI::MXNet::NDArray'))
    {
        return ([$args], 0);
    }
    elsif(blessed $args and $args->isa('AI::MXNet::Symbol'))
    {
        my $length = @{ $args->list_outputs() };
        $length = $length > 1 ? $length : 0;
        return ([$args], $length)
    }
    my @flat;
    my @fmts;
    for my $i (@{ $args })
    {
        my ($arg, $fmt) = __PACKAGE__->_flatten($i);
        push @flat, @{ $arg };
        push @fmts, $fmt;
    }
    return (\@flat, \@fmts);
}

method _regroup(
    $args, $fmt
)
{
    my $in_symbol = (blessed $args and $args->isa('AI::MXNet::Symbol'));
    my @ret;
    if(not ref $fmt)
    {
        my $len = @{$args} - 1;
        if($fmt == 0)
        {
            @ret = ([@{$args}[1..$len]]);
            if($in_symbol)
            {
                $ret[0] = AI::MXNet::Symbol->Group($ret[0]);
            }
            return (@{$args}[0], $ret[0]);
        }
        @ret = ([@{$args}[0..$fmt-1]], [@{$args}[$fmt..$len]]);
        if($in_symbol)
        {
            @ret = map { AI::MXNet::Symbol->Group($_) } @ret;
        }
        return @ret;
    }
    for my $i (@{ $fmt })
    {
        my $res;
        ($res, $args) = __PACKAGE__->_regroup($args, $i);
        push @ret, $res;
    }
    return (\@ret, $args);
}

has _prefix => (is => 'rw', init_arg => 'prefix', isa => 'Str');
has _params => (is => 'rw', init_arg => 'params', isa => 'Maybe[AI::MXNet::Gluon::ParameterDict]');
has [qw/_name _scope/] => (is => 'rw', init_arg => undef);
has [qw/_children/]    => (is => 'rw', init_arg => undef, default => sub { [] });
around BUILDARGS => \&AI::MXNet::Base::process_arguments;

sub AUTOLOAD {
    my $name = $AI::MXNet::Gluon::Block::AUTOLOAD;
    $name =~ s/.*:://;
    my $self = shift;
    AI::MXNet::Gluon::Mouse::has($name => (is => 'rw', 'init_arg' => undef, 'caller' => ref $self));
    $self->$name(@_);
}

sub BUILD
{
    my $self = shift;
    my ($prefix, $params) = AI::MXNet::Gluon::BlockScope->create($self->_prefix, $self->_params, $self->_alias);
    $self->_prefix($prefix);
    $self->_params($params);
    my $name = $prefix;
    $name =~ s/_$//;
    $self->_name($name);
    $self->_scope(AI::MXNet::Gluon::BlockScope->new(block => $self));
}

method _class_name()
{
    my $class = ref $self || $self;
    $class =~ s/^.+:://;
    $class;
}

method __setattr__($name, $current, $prev=)
{
    if(defined $prev)
    {
        if(
            (
                blessed $prev
                    and
                ($prev->isa('AI::MXNet::Gluon::Parameter') or $prev->isa('AI::MXNet::Gluon::Block'))
            )
            and not (blessed $current and (ref($prev) eq ref($current)))
        )
        {
            confess(
                sprintf(
                    "Changing attribute type for %s from %s to %s is not allowed.",
                    $self->name,
                    ref($prev),
                    ref($current)||'no ref'
                )
            );
        }
        if(blessed $current and $current->isa('AI::MXNet::Gluon::Block'))
        {
            for(my $i = 0; $i < @{ $self->_children }; $i++)
            {
                if(Scalar::Util::refaddr($self->_children->[$i]) eq Scalar::Util::refaddr($prev))
                {
                    $self->_children->[$i] = $current;
                }
            }
        }
    }
    if(blessed $current and $current->isa('AI::MXNet::Gluon::Block'))
    {
        $self->register_child($current);
    }
}

method _alias()
{
    lc $self->_class_name;
}

method attributes_hash()
{
    +{ map { $_ => $self->$_ } $self->meta->get_attribute_list };
}

use overload
    '""' => sub
    {
        my $self = shift;
        my $s = "%s(\n{%s}\n)";
        my @blocks;
        my %attributes_hash = %{ $self->attributes_hash };
        while(my ($k, $v) = each %attributes_hash)
        {
            if(blessed $v and $v->isa(__PACKAGE__))
            {
                push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            }
        }
        sprintf("%s(\n{%s}\n)", $self->_class_name, join("\n", @blocks));
    },
    '&{}' => sub { my $self = shift; sub { $self->call(@_) } };

method prefix()
{
    $self->_prefix;
}

method name()
{
    $self->_name;
}

method class()
{
    __PACKAGE__;
}

method name_scope(CodeRef $sub)
{
    $self->_scope->__enter__;
    $sub->();
    $self->_scope->__exit__;
}

=head2 params

        Returns this `Block`'s parameter dictionary (does not include its
        children's parameters).
=cut

method params()
{
    return $self->_params;
}

=head2 collect_params

        Returns a `ParameterDict` containing this `Block` and all of its
        children's Parameters.
=cut

method collect_params()
{
    my $ret = AI::MXNet::Gluon::ParameterDict->new(prefix => $self->_params->prefix);
    $ret->update($self->params);
    for my $cld (@{ $self->_children })
    {
        $ret->update($cld->collect_params());
    }
    return $ret;
}

=head2 save

        Save parameters to file.

        filename : str
            Path to file.
=cut

method save_params($filename)
{
    $self->collect_params->save($filename, $self->prefix);
}

=head2 load

        Load parameters from file.

        $filename : str
            Path to parameter file.
        :$ctx= : Context or list of Context
            Context(s) initialize loaded parameters on.
        :$allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        :$ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
=cut

method load_params(
    Str   $filename,
    Maybe [AI::MXNet::Context|ArrayRef[AI::MXNet::Context]] :$ctx=,
    Bool  :$allow_missing=0,
    Bool  :$ignore_extra=0
)
{
    $self->collect_params->load(
        $filename,
        ($ctx ? (ctx   => $ctx) : ()),
        allow_missing  => $allow_missing,
        ignore_extra   => $ignore_extra,
        restore_prefix => $self->prefix
    );
}

=head2 register_child

        Registers block as a child of self. `Block`s assigned to self as
        attributes will be registered automatically.
=cut

method register_child(AI::MXNet::Gluon::Block $block)
{
    push @{ $self->_children }, $block;
}

=head2 initialize

        Initializes `Parameter`s of this `Block` and its children.

        Equivalent to `block.collect_params().initialize(...)`
=cut

method initialize(
    Initializer $init=AI::MXNet::Initializer->Uniform(),
    AI::MXNet::Context|ArrayRef[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Bool :$verbose=0
)
{
    $self->collect_params->initialize(init => $init, ctx => $ctx, verbose => $verbose);
}


=head2 hybridize

        Activates or deactivates `HybridBlock`s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
=cut

method hybridize(Bool $active=1)
{
    $_->hybridize($active) for @{ $self->_children };
}

method call(@args)
{
    return $self->forward(@args);
}

=head2 forward

        Overrides to implement forward computation using `NDArray`. Only
        accepts positional arguments.

        Parameters
        ----------
        @args : array of NDArray
            Input tensors.
=cut

method forward(@args)
{
    confess("Not Implemented");
}

method register(Str $container)
{
    my $sub_name = $self->_class_name;
    no strict 'refs';
    *{$container.'_::'.$sub_name} = sub { shift; $self->new(@_) };
}

__PACKAGE__->register('AI::MXNet::Gluon');

package AI::MXNet::Gluon::HybridBlock;

=head2 NAME

    AI::MXNet::Gluon::HybridBlock

=head2 DESCRIPTION

    `HybridBlock` supports forwarding with both Symbol and NDArray.

    Forward computation in `HybridBlock` must be static to work with `Symbol`s,
    i.e. you cannot call `.asnumpy()`, `.shape`, `.dtype`, etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with `hybridize()`, `HybridBlock` works just like normal
    `Block`. After activation, `HybridBlock` will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of `hybrid_forward`.

    Refer `Hybrid tutorial <http://mxnet.io/tutorials/gluon/hybrid.html>`_ to see
    the end-to-end usage.
=cut

use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::Block';
has [qw/
        _reg_params _cached_graph
        _cached_op _cached_params
        _out_format _in_format
        _active _in_idx
/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->_reg_params({});
    $self->_cached_graph([]);
    $self->_active(0);
}

method __setattr__($name, $current, $prev=)
{
    $self->SUPER::__setattr__($name, $current, $prev);
    if(blessed $current and $current->isa('AI::MXNet::Gluon::Parameter'))
    {
        $self->_reg_params->{ $name } = $current;
    }
}

method register_child(AI::MXNet::Gluon::HybridBlock $block)
{
    push @{ $self->_children }, $block;
}

method hybridize(Bool $active=1)
{
    $self->_active($active);
    $self->SUPER::hybridize($active);
}

method _get_graph(@args)
{
    if(not @{ $self->_cached_graph })
    {
        my $args = [@args];
        my ($in_format, $out_format);
        ($args, $in_format) = __PACKAGE__->_flatten($args);
        $self->_in_format($in_format);
        my @inputs = map { AI::MXNet::Symbol->var("input_$_") } 0 .. @$args-1;
        my ($grouped_inputs) = __PACKAGE__->_regroup(\@inputs, $self->_in_format);
        my %params = map { $_ => $self->_reg_params->{$_}->var } keys %{ $self->_reg_params };
        my @out;
        $self->name_scope(sub {
            @out = $self->hybrid_forward('AI::MXNet::Symbol', @{ $grouped_inputs }, %params);
        });
        my $out = @out > 1 ? [@out] : $out[0];
        ($out, $out_format) = __PACKAGE__->_flatten($out);
        $self->_out_format($out_format);
        @{ $self->_cached_graph } = (\@inputs, AI::MXNet::Symbol->Group($out));
    }
    return @{ $self->_cached_graph };
}

=head2 infer_shape

        Infers shape of Parameters from inputs.
=cut

method infer_shape(@args)
{
    my ($inputs, $out) = $self->_get_graph(@args);
    my $args = \@args;
    ($args) = __PACKAGE__->_flatten($args);
    my %in;
    for(zip($inputs, $args)) {
        my ($i, $j) = @$_;
        $in{ $i->name } = $j->shape;
    }
    my ($arg_shapes, undef, $aux_shapes) = $out->infer_shape(%in);
    my %sdict;
    for(zip($out->list_arguments(), $arg_shapes)) {
        my ($i, $j) = @$_;
        $sdict{ $i } = $j;
    }
    my %aux;
    for(zip($out->list_auxiliary_states(), $aux_shapes)) {
        my ($i, $j) = @$_;
        $aux{ $i } = $j;
    }
    %sdict = (%sdict, %aux);
    for my $i ($self->collect_params->values)
    {
        $i->shape($sdict{ $i->name })
    }
}

method _build_cache(@args)
{
    my ($inputs, $out) = $self->_get_graph(@args);
    $self->_cached_op(AI::MXNet::NDArray->CachedOp($out));
    my %params = %{ $self->collect_params };
    $self->_cached_params([map { $params{ $_ } } @{ $out->list_inputs }]);
    assert(
        (
            ((keys %params) + (@{ $self->_cached_graph->[0] }))
                ==
            @{ $out->list_inputs }
        ),
        "Wrong number of inputs."
    );
    my %name2pos;
    enumerate(sub {
        my ($i, $var) = @_;
        $name2pos{ $var->name } = $i;
    }, $inputs);
    my @in_idx;
    enumerate(sub {
        my ($i, $name) = @_;
        if(not exists $params{ $name })
        {
            push @in_idx, [$i, $name2pos{ $name }];
        }
    }, $out->list_inputs);
    $self->_in_idx(\@in_idx);
}

use Data::Dumper;
method _call_cached_op(@args)
{
    if(not defined $self->_cached_op)
    {
        $self->_build_cache(@args);
    }

    my @cargs;
    eval {
        @cargs = map { defined($_) ? $_->data() : undef } @{ $self->_cached_params };
    };
    if($@)
    {
        if($@ =~ /DeferredInitializationError/)
        {
            $self->infer_shape(@args);
            map { $_->_finish_deferred_init if defined } @{ $self->_cached_params };
            @cargs = map { defined($_) ? $_->data() : undef } @{ $self->_cached_params };
        }
        else
        {
            confess($@);
        }
    }
    my $args = [@args];
    my $fmt;
    ($args, $fmt) = __PACKAGE__->_flatten($args);
    assert((Dumper($fmt) eq Dumper($self->_in_format)), "Invalid input format");
    for (@{ $self->_in_idx })
    {
        $cargs[$_->[0]] = $args->[$_->[1]];
    }
    my $out = $self->_cached_op->(@cargs);
    if(blessed $out and $out->isa('AI::MXNet::NDArray'))
    {
        $out = [$out];
    }
    my $ret = (__PACKAGE__->_regroup($out, $self->_out_format))[0];
    if(ref($ret) eq 'ARRAY' and wantarray)
    {
        return @$ret;
    }
    else
    {
        return $ret;
    }
}

=head2 forward

        Defines the forward computation. Arguments can be either
        `NDArray` or `Symbol`.
=cut

method forward($x, @args)
{
    if(blessed $x and $x->isa('AI::MXNet::NDArray'))
    {
        my @out;
        my $out;
        my $ctx = $x->context;
        my $current_ctx = AI::MXNet::Context->current_ctx;
        AI::MXNet::Context->set_current($ctx);
        if($self->_active)
        {
            if(wantarray)
            {
                my @out = $self->_call_cached_op($x, @args);
                AI::MXNet::Context->set_current($current_ctx);
                return @out;
            }
            else
            {
                my $out = $self->_call_cached_op($x, @args);
                AI::MXNet::Context->set_current($current_ctx);
                return $out;
            }
        }
        my %params;
        eval {
            %params = map { $_ => $self->_reg_params->{ $_ }->data($ctx) } keys %{ $self->_reg_params };
        };
        if($@)
        {
            if($@ =~ /DeferredInitializationError/)
            {
                $self->infer_shape($x, @args);
                $_->_finish_deferred_init for $self->collect_params->values;
                %params = map { $_ => $self->_reg_params->{ $_ }->data($ctx) } keys %{ $self->_reg_params };
            }
            else
            {
                confess($@);
            }
        }
        @out = $self->hybrid_forward('AI::MXNet::NDArray', $x, @args, %params);
        AI::MXNet::Context->set_current($current_ctx);
        return wantarray ? @out : $out[0];
    }
    assert(
        (blessed $x and $x->isa('AI::MXNet::Symbol')),
        "HybridBlock requires the first argument to forward be either ".
        "Symbol or NDArray, but got [".ref($x)."]"
    );
    my %params = map { $_ => $self->_reg_params->{ $_ }->var } keys %{ $self->_reg_params };
    my @ret;
    $self->name_scope(sub {
        @ret = $self->hybrid_forward('AI::MXNet::Symbol', $x, @args, %params);
    });
    return wantarray ? @ret : $ret[0];
}

=head2 hybrid_forward

        Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
=cut

method hybrid_forward($F, $x, @args)
{
    confess("NotImplementedError");
}

__PACKAGE__->register('AI::MXNet::Gluon');

package AI::MXNet::Gluon::SymbolBlock;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::SymbolBlock - Construct block from symbol.
=cut

=head1 DESCRIPTION

    Construct block from symbol. This is useful for using pre-trained models
    as feature extractors. For example, you may want to extract get the output
    from fc2 layer in AlexNet.

    Parameters
    ----------
    outputs : Symbol or list of Symbol
        The desired output for SymbolBlock.
    inputs : Symbol or list of Symbol
        The Variables in output's argument that should be used as inputs.
    params : ParameterDict
        Parameter dictionary for arguments and auxililary states of outputs
        that are not inputs.

    Examples
    --------
    >>> # To extract the feature from fc1 and fc2 layers of AlexNet:
    >>> alexnet = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=mx.cpu(),
                                                 prefix='model_')
    >>> inputs = mx.sym.var('data')
    >>> out = alexnet(inputs)
    >>> internals = out.get_internals()
    >>> print(internals.list_outputs())
    ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
    >>> outputs = [internals['model_dense0_relu_fwd_output'],
                   internals['model_dense1_relu_fwd_output']]
    >>> # Create SymbolBlock that shares parameters with alexnet
    >>> feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())
    >>> x = mx.nd.random_normal(shape=(16, 3, 224, 224))
    >>> print(feat_model(x))
=cut

has [qw/outputs inputs/] => (is => 'rw', isa => 'AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]');
method python_constructor_arguments() { [qw/outputs inputs/] }

sub BUILD
{
    my ($self, $orig_params) = @_;
    $self->_prefix('');
    $self->_params(AI::MXNet::Gluon::ParameterDict->new(prefix => '', shared => $orig_params->{params}));
    if(blessed $self->inputs and @{ $self->inputs->list_outputs } == 1)
    {
        $self->inputs([$self->inputs]);
    }
    if(blessed $self->outputs and @{ $self->outputs->list_outputs } == 1)
    {
        $self->outputs([$self->outputs]);
    }
    my ($syms, $in_format) = __PACKAGE__->_flatten($self->inputs);
    my ($out, $out_format) = __PACKAGE__->_flatten($self->outputs);
    $self->_in_format($in_format);
    $self->_out_format($out_format);
    $out = AI::MXNet::Symbol->Group($out);

    my %input_names;
    for my $i (@{ $syms })
    {
        assert(
            (@{ $i->get_internals->list_outputs() } == 1),
            "Input symbols must be variable, but $i is an output of operators"
        );
        $input_names{ $i->name } = 1;
    }

    for my $i (@{ $out->list_arguments })
    {
        if(not exists $input_names{$i})
        {
            $self->params->get($i, allow_deferred_init => 1);
        }
    }

    for my $i (@{ $out->list_auxiliary_states })
    {
        if(not exists $input_names{$i})
        {
            $self->params->get($i, grad_req => 'null', allow_deferred_init => 1);
        }
    }

    $self->_cached_graph([$syms, $out]);
    $self->_build_cache;
}

method forward($x, @args)
{
    if(blessed $x and $x->isa('AI::MXNet::NDArray'))
    {
        my @out;
        my $out;
        my $ctx = $x->context;
        my $current_ctx = AI::MXNet::Context->current_ctx;
        AI::MXNet::Context->set_current($ctx);
        if(wantarray)
        {
            my @out = $self->_call_cached_op($x, @args);
            AI::MXNet::Context->set_current($current_ctx);
            return @out;
        }
        else
        {
            my $out = $self->_call_cached_op($x, @args);
            AI::MXNet::Context->set_current($current_ctx);
            return $out;
        }
    }
    assert(
        (blessed $x and $x->isa('AI::MXNet::Symbol')),
        "HybridBlock requires the first argument to forward be either ".
        "Symbol or NDArray, but got [".ref($x)."]"
    );
    my $args = \@args;
    my $in_fmt;
    ($args, $in_fmt) = __PACKAGE__->_flatten([$x, @$args]);
    assert((Data::Dumper::Dumper($in_fmt) eq Data::Dumper::Dumper($self->_in_format)), "Invalid input format");
    my $ret = $self->_cached_graph->[1]->deepcopy;
    my %in;
    for(zip($self->_cached_graph->[0], $args)) {
        my ($k, $v) = @$_;
        $in{$k->name} = $v;
    }
    $ret->_compose(%in);
    $ret = (__PACKAGE__->_regroup($ret, $self->_out_format))[0];
    if(ref($ret) eq 'ARRAY' and wantarray)
    {
        return @$ret;
    }
    else
    {
        return $ret;
    }
}

method hybrid_forward(@args)
{
    confess('NotImplementedError');
}

__PACKAGE__->register('AI::MXNet::Gluon');

1;
