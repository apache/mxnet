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
has '_block'      => (is => 'ro', init_arg => 'block', weak_ref => 1);
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
    return $self if $self->_block->_empty_prefix;
    $self->_old_scope($_current);
    $_current = $self;
    $self->_name_scope(AI::MXNet::Symbol::NameManager->current);
    AI::MXNet::Symbol::NameManager->set_current(AI::MXNet::Symbol::Prefix->new(prefix => $self->_block->prefix));
    return $self;
}

method __exit__()
{
    return if $self->_block->_empty_prefix;
    AI::MXNet::Symbol::NameManager->set_current($self->_name_scope);
    $self->_name_scope(undef);
    $_current = $self->_old_scope;
}

package AI::MXNet::Gluon::Block;
use AI::MXNet::Gluon::Mouse;
use Scalar::Util qw(refaddr);

=head2 NAME

    AI::MXNet::Gluon::Block - Base class for all neural network layers and models.

=head2 DESCRIPTION

    Base class for all neural network layers and models. Your models should
    subclass this class.

    AI::MXNet::Gluon::Block can be nested recursively in a tree structure. You can create and
    assign child AI::MXNet::Gluon::Block as regular attributes

    use AI::MXNet::Gluon::NN qw(nn);
    use AI::MXNet qw(mx);

    package Model;
    use AI::MXNet::Gluon::Mouse;
    use AI::MXNet::Function::Parameters;
    extends 'AI::MXNet::Gluon::Block';

    sub BUILD
    {
        my $self = shift;
        $self->name_scope(sub {
            $self->dense0(nn->Dense(5, in_units=>5));
            $self->dense1(nn->Dense(5, in_units=>5));
        });
    }

    method forward($x)
    {
        return $self->dense1->($self->dense0->($x));
    }

    my $model = Model->new()
    $model->initialize(ctx=>mx->cpu(0))
    $model->(nd->zeros([10, 10], ctx=>mx->cpu(0)));


    Child AI::MXNet::Gluon::Block assigned this way will be registered and ->collect_params
    will collect their Parameters recursively.

    Parameters
    ----------
    Prefix acts like a name space. All children blocks created in parent block's
    name_scope will have parent block's prefix in their name.
    Please refer to
    naming tutorial https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/naming.html
    for more info on prefix and naming.

    params : AI::MXNet::Gluon::ParameterDict or undef
        AI::MXNet::Gluon::ParameterDict for sharing weights with the new AI::MXNet::Gluon::Block. For example,
        if you want `dense1` to share `dense0`'s weights, you can do

        $dense0 = nn->Dense(20);
        $dense1 = nn->Dense(20, params=>dense0->collect_params());
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
has [qw/_name _scope _empty_prefix/] => (is => 'rw', init_arg => undef);
has [qw/_children _forward_hooks _forward_pre_hooks/]  => (is => 'rw', init_arg => undef, default => sub { Hash::Ordered->new });
has '_reg_params' => (is => 'rw', init_arg => undef, default => sub { +{} });
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
    $self->_empty_prefix(defined $self->_prefix and $self->_prefix eq '');
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
    }
    if(blessed $current and $current->isa('AI::MXNet::Gluon::Block'))
    {
        $self->register_child($current, $name);
    }
    elsif(blessed $current and $current->isa('AI::MXNet::Gluon::Parameter'))
    {
        if(exists $self->_reg_params->{ $name })
        {
            confess("Overriding Parameter attribute $name is not allowed. ".
                "If you want to share parameters between blocks, please set".
                "'params' at Block construction instead."
            );
        }
        $self->_reg_params->{ $name } = $current;
    }
}

method _check_container_with_block()
{
    my $_find_unregistered_block_in_container;
    my %children = map { refaddr($_) => 1 } $self->_children->values;
    $_find_unregistered_block_in_container = sub { my ($data) = @_;
    # Find whether a nested container structure contains Blocks
        if(ref $data eq 'ARRAY')
        {
            for my $ele (@{ $data })
            {
                if($_find_unregistered_block_in_container->($ele))
                {
                    return 1
                }
            }
            return 0;
        }
        elsif(ref $data eq 'HASH')
        {
            for my $v (values %$data)
            {
                if($_find_unregistered_block_in_container->($v))
                {
                    return 1;
                }
            }
            return 0;
        }
        elsif(blessed $data and $data->isa('AI::MXNet::Gluon::Block'))
        {
            return not exists $children{ refaddr($data) };
        }
        else
        {
            return 0;
        }
    };
    my $attributes_hash = $self->attributes_hash();
    while(my ($k, $v) = each %{ $attributes_hash })
    {
        if((ref $v eq 'HASH' or ref $v eq 'ARRAY') and not $k =~ /^__/)
        {
            if($_find_unregistered_block_in_container->($v))
            {
                AI::MXNet::Logging->warning(
                    '"%s" is a unregsitered container with Blocks. '.
                    'Note that Blocks inside the list, tuple or dict will not be '.
                    'registered automatically. Make sure to register them using '.
                    'register_child() or switching to '.
                    'nn->Sequential/nn->HybridSequential instead. ',
                    $self->_class_name.'.'.$k
                );
            }
        }
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
        my $s = "%s(\n%s\n)";
        my @blocks;
        my %attributes_hash = %{ $self->attributes_hash };
        while(my ($k, $v) = each %attributes_hash)
        {
            if(blessed $v and $v->isa(__PACKAGE__))
            {
                push @blocks, "  ($k): ".AI::MXNet::Base::_indent("$v", 2);
            }
        }
        sprintf("%s(\n%s\n)", $self->_class_name, join("\n", @blocks));
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
    eval { $sub->(); };
    my $err = $@;
    $self->_scope->__exit__;
    confess($err) if $err;
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

        Returns a AI::MXNet::Gluon::ParameterDict containing this AI::MXNet::Gluon::Block and all of its
        children's Parameters(default), also can returns the ParameterDict
        with parameters that match a regular expression.

        For example, collects parameters specified in ['conv1_weight', 'conv1_bias', 'fc_weight',
        'fc_bias'

            $model->collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')

        or collects all parameters that have the name end with 'weight' or 'bias', this can be done
        using regular expressions.

            $model->collect_params('.*weight|.*bias')

=cut

method collect_params(Maybe[Str] $select=)
{
    $self->_check_container_with_block();
    my $ret = AI::MXNet::Gluon::ParameterDict->new(prefix => $self->_params->prefix);
    $ret->update($self->params, $select);
    for my $cld ($self->_children->values)
    {
        $ret->update($cld->collect_params($select));
    }
    return $ret;
}


method _collect_params_with_prefix(Str $prefix='')
{
    if($prefix)
    {
        $prefix .= '.';
    }
    my %ret = map { $prefix.$_ => $self->_reg_params->{ $_ } } keys %{ $self->_reg_params };
    my $iter = $self->_children->iterator;
    while(my ($name, $child) = $iter->())
    {
        %ret = (%ret, %{ $child->_collect_params_with_prefix("$prefix$name") });
    }
    return \%ret;
}

=head2 save_parameters

        Save parameters to file.

        filename : str
            Path to file.
=cut

method save_parameters(Str $filename)
{
    my $params = $self->_collect_params_with_prefix();
    my %arg_dict = map { $_ => $params->{$_}->_reduce } keys %{ $params };
    AI::MXNet::NDArray->save($filename, \%arg_dict);
}

=head2 load_parameters

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

method load_parameters(
    Str   $filename,
    AI::MXNet::Context|ArrayRef[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Bool  :$allow_missing=0,
    Bool  :$ignore_extra=0
)
{
    my $loaded = AI::MXNet::NDArray->load($filename);
    my $params = $self->_collect_params_with_prefix;
    return if not keys %$loaded and not keys %$params;

    if(not grep { /\./ } keys %$loaded)
    {
        # legacy loading
        %$loaded = ();
        $self->collect_params->load(
            $filename,
            ($ctx ? (ctx   => $ctx) : ()),
            allow_missing  => $allow_missing,
            ignore_extra   => $ignore_extra,
            restore_prefix => $self->prefix
        );
        return;
    }

    if(not $allow_missing)
    {
        for my $name (keys %$params)
        {
            if(not exists $loaded->{$name})
            {
                confess(
                    "Parameter $name is missing in file $filename, which contains parameters:".
                    join(',', keys %$loaded)."\n".
                    "Set allow_missing=>1 to ignore missing parameters."
                );
            }
        }
    }
    for my $name (keys %$loaded)
    {
        if(not $ignore_extra and not exists $params->{ $name })
        {
            confess(
                "Parameter $name loaded from file $filename is not present in ParameterDict, ".
                "which contains parameters ".
                join(',', keys %$params)."\n".
                "Set ignore_extra=>1 to ignore."
            );
        }
        $params->{$name}->_load_init($loaded->{$name}, $ctx) if exists $params->{$name};
    }
}

=head2 register_child

        Registers block as a child of self. `Block`s assigned to self as
        attributes will be registered automatically.
=cut

method register_child(AI::MXNet::Gluon::Block $block, Maybe[Str] $name=)
{
    $name //= $self->_children->keys;
    $self->_children->set($name, $block);
}

=head2 register_forward_pre_hook

        Registers a forward pre-hook on the block.

        The hook function is called immediately before 'forward'.
        It should not modify the input or output.

        Parameters
        ----------
        $hook : CodeRef or callable object
            The forward hook function of form $hook->($block, $input).

        Returns
        -------
        AI::MXNet::Gluon::Utils::HookHandle
=cut

method register_forward_pre_hook($hook)
{
    my $handle = AI::MXNet::Gluon::Utils::HookHandle->new;
    $handle->attach($self->_forward_pre_hooks, $hook);
    return $handle;
}

=head2 register_forward_hook

        Registers a forward hook on the block.

        The hook function is called immediately after 'forward'.
        It should not modify the input or output.

        Parameters
        ----------
        $hook : CodeRef or callable object
            The forward hook function of form $hook->($block, $input).

        Returns
        -------
        AI::MXNet::Gluon::Utils::HookHandle
=cut

method register_forward_hook($hook)
{
    my $handle = AI::MXNet::Gluon::Utils::HookHandle->new;
    $handle->attach($self->_forward_hooks, $hook);
    return $handle;
}

=head2 apply

        Applies $fn recursively to every child block as well as self.

        Parameters
        ----------
        $fn : callable
            Function to be applied to each submodule, of form `$fn->($block)`.

        Returns
        -------
        this block
=cut

method apply($fn)
{
    for my $cld ($self->_children->values)
    {
        $cld->apply($fn);
    }
    $fn->($self);
    return $self;
}

=head2 initialize


        Initializes AI::MXNet::Gluon::Parameters of this AI::MXNet::Gluon::Block and its children.
        Equivalent to $block->collect_params()->initialize(...)

        Parameters
        ----------
        $init : Initializer
            Global default Initializer to be used when Parameter->init is undefined`.
            Otherwise, Parameter->init takes precedence.
        ctx : Context or array ref of Context
            Keeps a copy of Parameters on one or many context(s).
        verbose : bool, default False
            Whether to verbosely print out details on initialization.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
=cut

method initialize(
    Initializer $init=AI::MXNet::Initializer->Uniform(),
    AI::MXNet::Context|ArrayRef[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Bool :$verbose=0,
    Bool :$force_reinit=0
)
{
    $self->collect_params->initialize(init => $init, ctx => $ctx, verbose => $verbose, force_reinit => $force_reinit);
}


=head2 hybridize

        Activates or deactivates `HybridBlock`s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        $active : bool, default True
            Whether to turn hybrid on or off.
        :$static_alloc : bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        :$static_shape : bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
=cut

method hybridize(
    Bool $active=1,
    %args
)
{
    $_->hybridize(
        $active,
        %args
    ) for $self->_children->values;
}

=head2 cast

        Cast this Block to use another data type.

        Parameters
        ----------
        dtype : Dtype
            The new data type.
=cut

method cast(Dtype $dtype)
{
    for my $child ($self->_children->values)
    {
        $child->cast($dtype);
    }
    $_->cast($dtype) for $self->params->values;
}

method call(@args)
{
    for my $hook ($self->_forward_pre_hooks->values)
    {
        $hook->($self, \@args);
    }
    my @out = $self->forward(@args);
    for my $hook ($self->_forward_hooks->values)
    {
        $hook->($self, \@args, \@out);
    }
    return wantarray ? @out : $out[0];
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
    my $dest = $self->can('new');
    my $func = sub {
        splice @_, 0, 1, $self;
        goto $dest;
    };
    no strict 'refs';
    *{"$container\::$sub_name"} = $func;
}

=head2 summary

        Print the summary of the model's output and parameters.

        The network must have been initialized, and must not have been hybridized.

        Parameters
        ----------
        @inputs : objects
            Any inputs that the model supports. For any tensor in the input, only
            AI::MXNet::NDArray is supported.
=cut

method summary(@inputs)
{
    my $summary = Hash::Ordered->new;
    my %seen;
    my @hooks;
    my $stringify;
    $stringify = sub {
        my $in = shift;
        if(ref($in) eq 'ARRAY')
        {
            return '('.join(', ', map { $stringify->($_) } @$in).')';
        }
         else
        {
            return "$in";
        }
    };
    my $_get_shape_str = sub { my ($args) = @_;
        $args = $args->[0] if(ref $args eq 'ARRAY' and @$args == 1);
        my ($flat_args, $fmts) = __PACKAGE__->_flatten($args);
        my $flat_arg_shapes = [map { (blessed($_) and $_->isa('AI::MXNet::NDArray')) ? $_->shape : $_ } @$flat_args];
        my $shapes = (__PACKAGE__->_regroup($flat_arg_shapes, $fmts))[0];
        my $shape_str = $stringify->($shapes);
        $shape_str =~ s/L//g;
        return $shape_str;
    };

    my $_register_summary_hook = sub { my ($block) = @_;
        unless(not $block->isa('AI::MXNet::Gluon:::HybridBlock') or not $block->_active)
        {
            confess("\"${\ $block->name }\" must not be hybridized to print summary.");
        }
        my $_summary_hook = sub { my ($block, undef, $outputs) = @_;
            my $class_name = $block->_class_name;
            my $block_idx = $summary->keys - 1;

            my $m_key = sprintf('%s-%i', $class_name, $block_idx+1);
            $summary->set($m_key, Hash::Ordered->new);
            $summary->get($m_key)->set('output_shape', $_get_shape_str->($outputs));

            my $params = 0;
            $summary->get($m_key)->set('trainable', 0);
            $summary->get($m_key)->set('shared', 0);
            for my $p (values %{ $block->_reg_params })
            {
                $params += $p->data->size;
                $summary->get($m_key)->set('trainable', $summary->get($m_key)->get('trainable') + ($p->grad_req eq 'null' ? 0 : $p->data->size));
                if(exists $seen{$p})
                {
                    $summary->get($m_key)->set('shared', $summary->get($m_key)->get('shared') + $p->data->size);
                }
                else
                {
                    $seen{$p} = 1;
                }
            }
            $summary->get($m_key)->set('n_params', $params);
        };

        if(not $block->isa('AI::MXNet::Gluon::NN::Sequential') and not $block->isa('AI::MXNet::Gluon::NN::HybridSequential'))
        {
            push @hooks, $block->register_forward_hook($_summary_hook);
        }
    };

    my $input = Hash::Ordered->new;
    $summary->set('Input', $input);
    $input->set('output_shape', $_get_shape_str->(\@inputs));
    $input->set('n_params', 0);
    $input->set('trainable', 0);
    $input->set('shared', 0);

    eval {
        $self->apply($_register_summary_hook);
        $self->(@inputs);

        my $line_format = "%20s  %42s %15s\n";
        print (('-')x80, "\n");
        printf($line_format, 'Layer (type)', 'Output Shape', 'Param #');
        print (('=')x80, "\n");
        my $total_params = 0;
        my $trainable_params = 0;
        my $shared_params = 0;
        for my $layer ($summary->keys)
        {
            printf($line_format, $layer, $summary->get($layer)->get('output_shape'), $summary->get($layer)->get('n_params'));
            $total_params += $summary->get($layer)->get('n_params');
            $trainable_params += $summary->get($layer)->get('trainable');
            $shared_params += $summary->get($layer)->get('shared');
        }
        print (('=')x80, "\n");
        print "Parameters in forward computation graph, duplicate included\n";
        print "   Total params: $total_params\n";
        print "   Non-trainable params: ", $total_params - $trainable_params, "\n";
        print "Shared params in forward computation graph: $shared_params\n";
        print "Unique parameters in model: ", $total_params - $shared_params, "\n";
        print (('-')x80, "\n");
    };
    $_->detach for @hooks;
}

__PACKAGE__->register('AI::MXNet::Gluon');

package AI::MXNet::Gluon::HybridBlock;
=head2 NAME

    AI::MXNet::Gluon::HybridBlock

=head2 DESCRIPTION

    HybridBlock supports forwarding with both Symbol and NDArray.

    Forward computation in HybridBlock must be static to work with Symbols,
    i.e. you cannot call aspdl, shape, dtype, etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with hybridize(), HybridBlock works just like normal
    Block. After activation, HybridBlock will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of hybrid_forward.

    Refer Hybrid tutorial L<https://mxnet.io/tutorials/gluon/hybrid.html> to see
    the end-to-end usage.
=cut

use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::Block';
has [qw/
        _cached_graph
        _cached_op
        _out_format _in_format
        _active _flags _cached_op_args
/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->_active(0);
    $self->_flags([]);
    $self->_cached_graph([]);
    $self->_cached_op_args([]);
}

method __setattr__($name, $current, $prev=)
{
    $self->SUPER::__setattr__($name, $current, $prev);
    if(blessed $current and $current->isa('AI::MXNet::Gluon::HybridBlock'))
    {
        $self->_clear_cached_op();
    }
}

method register_child(AI::MXNet::Gluon::HybridBlock $block, Maybe[Str] $name=)
{
    $self->SUPER::register_child($block, $name);
    $self->_clear_cached_op();
}

method hybridize(@args)
{
    my $active;
    if(@args%2)
    {
        $active = shift(@args);
    }
    else
    {
        $active = 1;
    }
    $self->_active($active);
    @{ $self->_flags } = @args;
    $self->_clear_cached_op();
    if($self->_active and ($self->_forward_hooks or $self->_forward_pre_hooks))
    {
        AI::MXNet::Logging->warning(
            "$self is being hybridized while still having forward hook/pre-hook. ".
            "If $self is a child of HybridBlock, the hooks will not take effect."
        );
    }
    $self->SUPER::hybridize($self->_active, @args);
}

method cast(Dtype $dtype)
{
    $self->_clear_cached_op;
    $self->SUPER::cast($dtype);
}

method  _infer_attrs($infer_fn, $attr, @args)
{
    my ($inputs, $out) = $self->_get_graph(@args);
    my ($args) = __PACKAGE__->_flatten([@args]);
    my %in;
    zip(sub {
        my ($i, $j) = @_;
        $in{ $i->name } = $j->$attr;
    }, $inputs, $args);
    my ($arg_attrs, $aux_attrs);
    ($arg_attrs, undef, $aux_attrs) = $out->$infer_fn(%in);
    if(not defined $arg_attrs)
    {
        confess($@);
    }
    my %sdict;
    zip(sub {
        my ($i, $j) = @_;
        $sdict{ $i } = $j;
    }, $out->list_arguments, $arg_attrs);
    zip(sub {
        my ($i, $j) = @_;
        $sdict{ $i } = $j;
    }, $out->list_auxiliary_states, $aux_attrs);

    for my $i ($self->collect_params->values)
    {
        $i->$attr($sdict{ $i->name });
    }
}

method infer_shape(@args)
{
    $self->_infer_attrs('infer_shape', 'shape', @args);
}

method infer_type(@args)
{
    $self->_infer_attrs('infer_type', 'dtype', @args);
}

method _get_graph(@args)
{
    if(not @{ $self->_cached_graph })
    {
        my $args = [@args];
        my ($in_format, $out_format);
        ($args, $in_format) = __PACKAGE__->_flatten($args);
        $self->_in_format($in_format);
        my @inputs; 
        if(@args > 1)
        {
            @inputs = map { AI::MXNet::Symbol->var("data_$_") } 0 .. @$args-1;
        }
        else
        {
            @inputs = (AI::MXNet::Symbol->var("data"))
        }
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

method _build_cache(@args)
{
    my ($data, $out) = $self->_get_graph(@args);
    my $i = 0;
    my %data_names = map { $_->name => $i++ } @{ $data };
    my $params = $self->collect_params;
    my $input_names = $out->list_inputs;
    my %param_names = map { $_ => 1 } $params->keys;
    my %expected_names = map { $_ => 1 } @{ $input_names };
    for my $name (keys %expected_names)
    {
        assert(
            (exists $param_names{ $name } or exists $data_names{ $name }),
            "Unknown input to HybridBlock: $name"
        );
    }
    my $unused = join(', ', map { "$data_names{$_}-th" } grep { !exists $expected_names{ $_ } } keys %data_names);
    AI::MXNet::Logging->warn(
        "The $unused input to HybridBlock is not used by any ".
        "computation. Is this intended?"
    ) if $unused;
    $unused = join(', ', grep { !exists $expected_names{ $_ } } keys %param_names);
    AI::MXNet::Logging->warn(
        "Parameter %s is not used by any computation. " .
        "Is this intended?"
    ) if $unused;

    my @data_indices;
    my @param_indices;
    $self->_cached_op_args([]);
    enumerate(sub {
        my ($i, $name) = @_;
        if(exists $data_names{ $name })
        {
            push @data_indices, $i;
            push @{ $self->_cached_op_args }, [1, $data_names{$name}];
        }
        else
        {
            push @param_indices, $i;
            push @{ $self->_cached_op_args }, [0, $params->params->get($name)];
        }
    }, $input_names);
    my %flags = (
        data_indices  => \@data_indices,
        param_indices => \@param_indices,
        @{ $self->_flags }
    );
    $self->_cached_op(AI::MXNet::CachedOp->new($out, \%flags));
}

method _deferred_infer_shape(@args)
{
    eval {
        $self->infer_shape(@args)
    };
    if($@)
    {
        confess(
            "Deferred initialization failed because shape".
            " cannot be inferred. $@"
        );
    }
}

method _clear_cached_op()
{
    $self->_cached_graph([]);
    $self->_cached_op(undef);
}

use Data::Dumper;
method _call_cached_op(@args)
{
    if(not defined $self->_cached_op)
    {
        $self->_build_cache(@args);
    }
    my $args = [@args];
    my $fmt;
    ($args, $fmt) = __PACKAGE__->_flatten($args);
    assert((Dumper($fmt) eq Dumper($self->_in_format)), "Invalid input format");
    my @cargs;
    eval {
        @cargs = map { (not $_->[0]) ? $_->[1]->data() : $args->[$_->[1]] } @{ $self->_cached_op_args };
    };
    if($@)
    {
        if($@ =~ /DeferredInitializationError/)
        {
            $self->_deferred_infer_shape(@$args);
            @cargs = ();
            map {
                if($_->[0])
                {
                    push @cargs, $args->[$_->[1]];
                }
                else
                {
                    $_->[1]->_finish_deferred_init();
                    push @cargs, $_->[1]->data;
                }
            } @{ $self->_cached_op_args };
        }
        else
        {
            confess($@);
        }
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
        NDArray or Symbol
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
                $self->_deferred_infer_shape($x, @args);
                $_->_finish_deferred_init for $self->params->values;
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

=head2 export

        Export HybridBlock to json format that can be loaded by AI::MXNet::Module
        or the C++ interface.

        When there are only one input, it will have name 'data'. When there
        Are more than one inputs, they will be named as 'data0', 'data1', etc.

        Parameters
        ----------
        $path : str
            Path to save model. Two files 'path-symbol.json' and 'path-xxxx.params'
            will be created, where xxxx is the 4 digits epoch number.
        :$epoch=0 : Int
            Epoch number of saved model.
=cut

method export(Str $path, :$epoch=0)
{
    if(not @{ $self->_cached_graph })
    {
        confess(
            "Please first call \$block->hybridize() and then run forward with ".
            "this block at least once before calling export."
        );
    }
    my $sym = $self->_cached_graph->[1];
    $sym->save("$path-symbol.json");

    my %arg_names = map { $_ => 1 } @{ $sym->list_arguments };
    my %aux_names = map { $_ => 1 } @{ $sym->list_auxiliary_states };
    my %arg_dict;
    my $params = $self->collect_params;
    for my $name ($params->keys)
    {
        my $param = $params->get($name);
        if(exists $arg_names{ $name })
        {
            $arg_dict{ "arg:$name" } = $param->_reduce;
        }
        else
        {
            assert(exists $aux_names{ $name });
            $arg_dict{ "aux:$name" } = $param->_reduce;
        }
    }
    AI::MXNet::NDArray->save(sprintf('%s-%04d.params', $path, $epoch), \%arg_dict);
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
    >>> # To extract the feature from fc1 and fc2 layers of AlexNet
    >>> $alexnet = gluon->model_zoo->vision->alexnet(pretrained=>1, ctx=>mx->cpu(),
                                                 prefix=>'model_');
    >>> $inputs = mx->sym->var('data');
    >>> $out = $alexnet->($inputs);
    >>> $internals = $out->get_internals()
    >>> print($internals->list_outputs())
    ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
    >>> $outputs = [$internals->slice('model_dense0_relu_fwd_output'),
                   $internals->slice('model_dense1_relu_fwd_output')];
    >>> # Create SymbolBlock that shares parameters with alexnet
    >>> $feat_model = gluon->SymbolBlock($outputs, $inputs, params=>$alexnet->collect_params());
    >>> $x = mx->nd->random_normal(shape=>[16, 3, 224, 224]);
    >>> print($feat_model->($x));
=cut

has [qw/outputs inputs/] => (is => 'rw', isa => 'AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]');
method python_constructor_arguments() { [qw/outputs inputs/] }

sub BUILD
{
    my ($self, $orig_params) = @_;
    return unless defined $self->outputs and defined $self->inputs;
    $self->_prefix('');
    $self->_params(AI::MXNet::Gluon::ParameterDict->new(prefix => '', shared => $orig_params->{params}));
    if(blessed $self->inputs and @{ $self->inputs->list_outputs } == 1)
    {
        $self->inputs([$self->inputs]);
    }
    if(not blessed $self->outputs and @{ $self->outputs } == 1)
    {
        $self->outputs($self->outputs->[0]);
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

    # check if any symbol is row_sparse
    my $row_sparse_storage = STORAGE_TYPE_STR_TO_ID->{row_sparse};
    for my $i (@{ $out })
    {
        for my $j (@{ $i->get_internals })
        {
            assert(
                (not defined $j->attr("__storage_type__") or $j->attr("__storage_type__") ne $row_sparse_storage),
                "SymbolBlock doesn't support Parameter ${\ $j->name }  because its storage ".
                "type is 'row_sparse'."
            );
        }
    }

    my $arg_params = $out->list_arguments;
    my $aux_params = $out->list_auxiliary_states;
    my ($arg_types, $aux_types) = _infer_param_types($syms, $out, $arg_params, $aux_params);

    for(enumerate($arg_params))
    {
        my ($i, $arg) = @$_;
        if(not exists $input_names{ $arg })
        {
            $self->params->get($arg, allow_deferred_init => 1, dtype => $arg_types->[$i]);
        }
    }

    for(enumerate($aux_params))
    {
        my ($i, $arg) = @$_;
        if(not exists $input_names{ $arg })
        {
            $self->params->get($arg, grad_req => 'null', allow_deferred_init => 1, dtype => $aux_types->[$i]);
        }
    }

    $self->_cached_graph([$syms, $out]);
    my $prefix = _common_prefix($self->_params->keys);
    my %params = $self->_params->items;
    while(my ($key, $val) = each %params)
    {
        $key =~ s/^$prefix//;
        $self->_reg_params->{ $key } = $val;
    }
    $self->_prefix($prefix);
}


func _infer_param_types($in_params, $out_params, $arg_params, $aux_params, $default_dtype='float32')
{
    # Utility function that helps in inferring DType of args and auxs params
    # from given input param.
    # Parameters
    # ----------
    # in_params: array ref of AI::MXNet::Symbol objects
    #     List of input symbol variables.
    # out_params: AI::MXNet::Symbol
    #     Output symbol variable.
    # arg_params: array ref of Str
    #     List of names of argument parametrs.
    # aux_params: array ref of Str
    #     List of names of auxiliary parameters.
    # default_dtype: Dtype, default 'float32'
    #     Default data type for arg_params and aux_params, if unable to infer the type.
    #  Returns
    # -------
    # arg_types: Array ref of Dtype
    #     List of arg_params type. Order is same as arg_params.
    #     Defaults to 'float32', if unable to infer type.
    # aux_types: Array ref of Dtype
    #     List of aux_params type. Order is same as aux_params.
    #     Defaults to 'float32', if unable to infer type.

    my $arg_types;
    my $aux_types;
    # Get Input symbol details. This will be used to infer types of
    # other parameters.
    my @input_sym_names = map { $_->name } @{ $in_params };
    # Try to infer input types. If not successful, we will set default dtype.
    # If successful, we will try to infer other params in the graph.
    my @input_sym_arg_types;
    my $can_infer_input_type = 1;
    for my $in_param(@{ $in_params })
    {
        my $input_sym_arg_type = ($in_param->infer_type)[0];
        if(not $input_sym_arg_type or @$input_sym_arg_type < 1)
        {
            $can_infer_input_type = 0;
            last;
        }
        else
        {
            push @input_sym_arg_types, $input_sym_arg_type->[0];
        }
    }
    # Try to infer types of other parameters.
    if($can_infer_input_type)
    {
        my %params = map { $_->[0] => $_->[1] } zip(\@input_sym_names, \@input_sym_arg_types);
        ($arg_types, undef, $aux_types) = $out_params->infer_type(%params);
        if(not defined $arg_types or @$arg_types != @$arg_params)
        {
            $arg_types = [($default_dtype)x@$arg_params];
        }
        if(not defined $aux_types or @$aux_types != @$aux_params)
        {
            $aux_types = [($default_dtype)x@$aux_params];
        }
    }
    return ($arg_types, $aux_types);
}

func _common_prefix(@names)
{
    if(not @names)
    {
        return ''
    }
    my $prefix = $names[0];
    for my $name (@names)
    {
        my $i = 0;
        while($i < length($prefix) and $i < length($name) and substr($prefix, $i, 1) eq substr($name, $i, 1))
        {
            $i++;
        }
        $prefix = substr($prefix, 0, $i);
    }
    return $prefix;
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

method _clear_cached_op()
{
    my $tmp = $self->_cached_graph;
    $self->SUPER::_clear_cached_op;
    $self->_cached_graph($tmp);
}

method hybrid_forward(@args)
{
    confess('NotImplementedError');
}

=head2 imports

        Import model previously saved by HybridBlock->export or
        Module->save_checkpoint as a SymbolBlock for use in Gluon.

        Parameters
        ----------
        $symbol_file : Str
            Path to symbol file.
        $input_names : Str|ArrayRef[Str]
            List of input variable names
        :$param_file : Str, optional
            Path to parameter file.
        $ctx : Context, default undef
            The context to initialize SymbolBlock on.

        Returns
        -------
        SymbolBlock
            SymbolBlock loaded from symbol and parameter files.
=cut

method imports(Str $symbol_file, Str|ArrayRef[Str] $input_names, Maybe [Str] $param_file=, Maybe[AI::MXNet::Context] $ctx=)
{
    my $sym = AI::MXNet::Symbol->load($symbol_file);
    $input_names = [$input_names] unless ref $input_names;
    my @inputs = map { AI::MXNet::Symbol->var($_) } @{ $input_names };
    my $ret = __PACKAGE__->new($sym, \@inputs);
    if(defined $param_file)
    {
        $ret->load_parameters($param_file, (defined $ctx ? (ctx=>$ctx) : ()));
    }
    return $ret
}

__PACKAGE__->register('AI::MXNet::Gluon');

1;
