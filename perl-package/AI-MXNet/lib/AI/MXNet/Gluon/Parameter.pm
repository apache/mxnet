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
use Hash::Ordered;
package AI::MXNet::Gluon::Parameter;
use AI::MXNet::NS;
use AI::MXNet::Function::Parameters;

=head1 NAME 

    AI::MXNet::Gluon::Parameter - A Container holding parameters (weights) of AI::MXNEt::Gluon::Block(s).
=cut

=head1 DESCRIPTION

    AI::MXNet::Gluon::Parameter holds a copy of the parameter on each AI::MXNet::Context after
    it is initialized with AI::MXNet::Gluon::Parameter->initialize(...)`. If grad_req is
    not 'null', it will also hold a gradient array on each AI::MXNet::Context

        $ctx = mx->gpu(0);
        $x = mx->nd->zeros([16, 100], ctx=>$ctx);
        $w = mx->gluon->Parameter('fc_weight', shape=>[64, 100], init=>mx->init->Xavier());
        $b = mx->gluon->Parameter('fc_bias', shape=>[64], init=>mx->init->Zero());
        $w->initialize(ctx=>$ctx);
        $b->initialize(ctx=>ctx);
        $out = mx->nd->FullyConnected($x, $w->data($ctx), $b->data($ctx), num_hidden=>64);

    Parameters
    ----------
    name : str
        Name of this parameter.
    grad_req : {'write', 'add', 'null'}, default 'write'
        Specifies how to update gradient to grad arrays.

        - 'write' means everytime gradient is written to grad NDArray.
        - 'add' means everytime gradient is added to the grad NDArray. You need
          to manually call zero_grad() to clear the gradient buffer before each
          iteration when using this option.
        - 'null' means gradient is not requested for this parameter. gradient arrays
          will not be allocated.
    shape : array ref of int or int, default undef
        Shape of this parameter. By default shape is not specified. Parameter with
        unknown shape can be used for `Symbol` API, but `init` will throw an error
        when using `NDArray` API.
    dtype : Dtype, default 'float32'
        Data type of this parameter. For example, 'float64'.
    lr_mult : float, default 1.0
        Learning rate multiplier. Learning rate will be multiplied by lr_mult
        when updating this parameter with optimizer.
    wd_mult : float, default 1.0
        Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.
    init : Initializer, default None
        Initializer of this parameter. Will use the global initializer by default.
    stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter.
    grad_stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter's gradient.


    Attributes
    ----------
    grad_req : {'write', 'add', 'null'}
        This can be set before or after initialization. Setting grad_req to null
        with $x->grad_req = 'null' saves memory and computation when you don't
        need gradient w.r.t x.
=cut

use Mouse;
use AI::MXNet::Base;
use overload '""' => sub {
        my $self = shift;
        "Parameter " . $self->name.
        " (shape=(" . join(', ', @{ $self->shape//[] }) .")".
        ", dtype=" . $self->dtype.
        ", stype=" . $self->stype.")"
    },
    fallback => 1;

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    if(@_ % 2)
    {
        my $name = shift;
        return $class->$orig(name => $name, @_);
    }
    else
    {
        return $class->$orig(@_);
    }
};

sub BUILD
{
    my $self = shift;
    $self->grad_req($self->_grad_req);
    $self->_shape([$self->_shape]) if defined $self->_shape and not ref $self->_shape;
    $self->_deferred_init([]);
}

has 'name'                => (is => 'ro', isa => 'Str', required => 1);
has '_grad_req'           => (is => 'rw', isa => 'GradReq', init_arg => 'grad_req', default => 'write');
has '_shape'              => (is => 'rw', isa => 'Maybe[Shape|Int]', init_arg => 'shape');
has 'dtype'               => (is => 'rw', isa => 'Dtype', default => 'float32');
has ['stype',
     'grad_stype']        => (is => 'rw', isa => 'Stype', default => 'default');
has [qw/lr_mult wd_mult/] => (is => 'rw', isa => 'Num', default => 1);
has 'init'                => (is => 'rw', isa => 'Maybe[Initializer]');
has 'allow_deferred_init' => (is => 'rw', isa => 'Bool', default => 0);
has 'differentiable'      => (is => 'rw', isa => 'Bool', default => 1);
has [qw/_var _data _grad
    _deferred_init _trainer
    _ctx_list _ctx_map/]  => (is => 'rw', init_arg => undef);

method grad_req(Maybe[GradReq] $req=)
{
    return $self->_grad_req unless defined $req;
    if(not $self->differentiable)
    {
        $req = 'null';
    }
    return if $self->_grad_req eq $req;
    $self->_grad_req($req);
    if($req eq 'null' and defined $self->_grad)
    {
        $self->_grad(undef);
        $self->_data([map { $_->detach } @{ $self->_data }]);
    }
    elsif(defined $self->_data)
    {
        $self->_init_grad();
    }
}

method shape(@args)
{
    return $self->_shape unless @args;
    if(not defined $args[0])
    {
        $self->_shape(undef);
        return undef;
    }
    if(not defined $self->_shape and defined $args[0])
    {
        $self->_shape(ref $args[0] ? $args[0] : [$args[0]]);
        return $self->_shape;
    }
    my $new_shape = ref $args[0] ? $args[0] : [$args[0]];
    my $shape_validated = 0;
    if(@{ $self->_shape } == @{ $new_shape })
    {
        $shape_validated = 1;
        zip(sub {
            my ($i, $j) = @_;
            return unless $i;
            return if $i == $j;
            $shape_validated = 0;
        }, $self->_shape, $new_shape);
    }
    assert($shape_validated, 'Expected shape is incompatible with given shape');
    $self->_shape($new_shape);
    return $self->_shape;
}

method _set_trainer($trainer)
{
    if($self->stype ne 'default' and $self->_trainer and $trainer and Scalar::Util::refaddr($self->_trainer) ne Scalar::Util::refaddr($trainer))
    {
        confess(
            "Failed to set the trainer for Parameter '${\ $self->name }' because it was already set. ".
            "More than one trainers for a ${\ $self->stype } Parameter is not supported."
        );
    }
    $self->_trainer($trainer);
}

method _get_row_sparse($arr_list, $ctx, AI::MXNet::NDArray $row_id)
{
    if(not $self->_trainer)
    {
        confess(
            "Cannot get row_sparse data for Parameter '${\ $self->name }' when no ".
            "Trainer is created with it."
        );
    }
    my $results = $self->_check_and_get($arr_list, $ctx);

    # fetch row sparse params from the trainer
    $self->_trainer->_row_sparse_pull($self, $results, $row_id);
    return $results;
}

method _check_and_get($arr_list, $ctx)
{
    if(defined $arr_list)
    {
        if(ref $ctx eq 'ARRAY')
        {
            return $arr_list;
        }
        if(not defined $ctx)
        {
            if(@{ $arr_list } == 1)
            {
                return $arr_list->[0];
            }
            else
            {
                $ctx = AI::MXNet::Context->current_ctx;
            }
        }
        my $ctx_list = $self->_ctx_map->[$ctx->device_type_id&1];
        if($ctx->device_id < @{ $ctx_list })
        {
            my $idx = $ctx_list->[$ctx->device_id];
            if(defined $idx)
            {
                return $arr_list->[$idx];
            }
        }
        confess(
            "Parameter '${\ $self->name }' was not initialized on context $ctx. ".
            "It was only initialized on @{ $self->_ctx_list }."
        );
    }
    if(@{ $self->_deferred_init })
    {
        confess("DeferredInitializationError: ".
            "Parameter '${\ $self->name }' has not been initialized yet because initialization was ".
            "deferred. Actual initialization happens during the first forward pass. ".
            "Please pass one batch of data through the network before accessing Parameters. ".
            "You can also avoid deferred initialization by specifying in_units, ".
            "num_features, etc., for network layers.");
    }
    confess(
        "Parameter '${\ $self->name }' has not been initialized. Note that ".
        "you should initialize parameters and create Trainer ".
        "with Block.collect_params() instead of Block.params ".
        "because the later does not include Parameters of ".
        "nested child Blocks"
    );
}


# (Re)initializes by loading from data. 
method _load_init($data, $ctx)
{
    if($self->shape)
    {
        for(zip($self->shape, $data->shape)) {
            my ($self_dim, $data_dim) = @$_;
            assert(
                ($self_dim == 0 or $self_dim == $data_dim),
                sprintf(
                    "Failed loading Parameter '%s' from saved params: ".
                    "shape incompatible expected (%s) vs saved (%s)",
                    $self->name, "@{$self->shape}", "@{$data->shape}"
                )
            );
        }
        $self->shape([map { $_->[0] ? $_->[0] : $_->[1] } zip($self->shape, $data->shape)]);
    }
    if($self->dtype)
    {
        assert(
            ($self->dtype eq $data->dtype),
            sprintf(
                "Failed loading Parameter '%s' from saved params: ".
                "dtype incompatible expected %s vs saved %s",
                $self->name, $self->dtype, $data->dtype
            )
        );
    }
    if($self->stype ne $data->stype)
    {
        $data = $data->tostype($self->stype);
    }

    if(blessed ($ctx) and $ctx->isa('AI::MXNet::Context'))
    {
        $ctx = [$ctx];
    }
    if(not defined $self->_data)
    {
        if(@{ $self->_deferred_init })
        {
            assert(
                (not defined $ctx or join('', @{ $ctx }) eq join('', @{ $self->_deferred_init->[1] })),
                sprintf(
                    "Failed to load Parameter '%s' on %s because it was ".
                    "previously initialized on %s.",
                    $self->name, $ctx, $self->list_ctx
                )
            );
            $ctx = $self->_deferred_init->[1];
        }
        elsif(not defined $ctx)
        {
            $ctx = [AI::MXNet::Context->cpu];
        }
        $self->_init_impl($data, $ctx);
    }
    else
    {
        assert(
            (not defined $ctx or join('', @{ $ctx }) eq join('', @{ $self->list_ctx })),
            sprintf(
                "Failed to load Parameter '%s' on %s because it was ".
                "previously initialized on %s.",
                $self->name, "@$ctx", "@{$self->list_ctx}"
            )
        );
        $self->set_data($data);
    }
    $self->_deferred_init([]);
}

# Finishes deferred initialization.
method _finish_deferred_init()
{
    return unless @{ $self->_deferred_init };
    my ($init, $ctx, $default_init, $data) = @{ $self->_deferred_init };
    $self->_deferred_init([]);
    assert(
        (defined($self->shape) and product(@{ $self->shape }) > 0),
        sprintf(
            "Cannot initialize Parameter '%s' because it has ".
            "invalid shape: %s. Please specify in_units, ".
            "in_channels, etc for `Block`s.",
            $self->name, $self->shape
        )
    );
    AI::MXNet::AutoGrad->pause(sub {
        if(not defined $data)
        {
            $data = AI::MXNet::NDArray->zeros(
                $self->shape,
                dtype => $self->dtype,
                ctx => AI::MXNet::Context->cpu,
                stype => $self->stype
            );
            AI::MXNet::Initializer->new->(
                AI::MXNet::InitDesc->new(
                    name => $self->name,
                    attrs => { __init__ => defined $init ? "$init" : "$default_init" }
                ),
                $data
            );
        }
        $self->_init_impl($data, $ctx);
    });
}

# Sets data and grad.
method _init_impl($data, $ctx_list)
{
    $self->_ctx_list([@{ $ctx_list }]);
    $self->_ctx_map([[], []]);
    enumerate(sub {
        my ($i, $ctx) = @_;
        my $dev_list = $self->_ctx_map->[$ctx->device_type_id&1];
        while(@{ $dev_list } <= $ctx->device_id)
        {
            push @{ $dev_list }, undef;
        }
        $dev_list->[$ctx->device_id] = $i;
    }, $self->_ctx_list);
    $self->_data([map { $data->copyto($_) } @{ $self->_ctx_list }]);
    $self->_init_grad;
}

# Initialize grad buffers.
method _init_grad()
{
    if($self->grad_req eq 'null')
    {
        $self->_grad(undef);
        return;
    }
    $self->_grad([
        map {
            AI::MXNet::NDArray->zeros(
                $_->shape, dtype => $_->dtype,
                ctx => $_->context, stype => $self->grad_stype
            )
        } @{ $self->_data }
    ]);
    AI::MXNet::AutoGrad->mark_variables(
        $self->_check_and_get($self->_data, []),
        $self->_grad,
        grad_reqs => $self->grad_req
    );
}

# Reduce data from multiple contexts to cpu.
method _reduce()
{
    my $data;
    my $ctx = AI::MXNet::Context->cpu;
    if($self->stype eq 'default')
    {
        my $block = $self->list_data;
        $data = AI::MXNet::NDArray->add_n(map { $_->copyto($ctx) } @{ $block }) / @{ $block };
    }
    else
    {
        my $all_row_ids = AI::MXNet::NDArray->arange(stop => $self->shape->[0], dtype=>'int64', ctx=>$ctx);
        $data = AI::MXNet::NDArray->zeros($self->shape, stype=>'row_sparse', ctx=>$ctx);
        $self->_trainer->_row_sparse_pull($self, $data, $all_row_ids, 1);
    }
    return $data;
}

=head2 initialize

        Initializes parameter and gradient arrays. Only used for `NDArray` API.

        Parameters
        ----------
        :$init : Initializer
            The initializer to use. Overrides AI::MXNet::Gluon::Parameter->init and default_init.
        :$ctx : AI::MXNet::Context or array ref of AI::MXNet::Context, defaults to AI::MXNet::Context->current_ctx().
            Initialize Parameter on given context. If ctx is a list of Context, a
            copy will be made for each context.
            Copies are independent arrays. User is responsible for keeping
            their values consistent when updating. Normally gluon->Trainer does this for you.
        :$default_init : Initializer
            Default initializer is used when both 'init' and AI::MXNet::Gluon::Parameter->init are undefined.
        :$force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.

        Examples
        --------
        >>> $weight = mx->gluon->Parameter('weight', shape=>[2, 2]);
        >>> $weight->initialize(ctx=>mx->cpu(0));
        >>> print $weight->data
        [[-0.01068833  0.01729892]
         [ 0.02042518 -0.01618656]]
        <NDArray 2x2 @cpu(0)>
        >>> print $weight->grad()
        [[ 0.  0.]
         [ 0.  0.]]
        <NDArray 2x2 @cpu(0)>
        >>> $weight->initialize(ctx=>[mx->gpu(0), mx->gpu(1)]);
        >>> print $weight->data(mx->gpu(0));
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(0)>
        >>> print $weight->data(mx->gpu(1))
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(1)>
=cut

method initialize(
    Maybe[Initializer]                                     :$init=,
    Maybe[AI::MXNet::Context|ArrayRef[AI::MXNet::Context]] :$ctx=AI::MXNet::Context->current_ctx,
    Initializer                                            :$default_init=AI::MXNet::Initializer->Uniform,
    Bool                                                   :$force_reinit=0
)
{
    $ctx //=AI::MXNet::Context->current_ctx;
    if(defined $self->_data and not $force_reinit)
    {
        AI::MXNet::Logging->warning(
            "Parameter '%s' is already initialized, ignoring. ".
            "Set force_reinit=True to re-initialize.", $self->name
        );
        return;
    }
    $self->_data(undef);
    $self->_grad(undef);
    if(blessed($ctx) and $ctx->isa('AI::MXNet::Context'))
    {
        $ctx = [$ctx];
    }
    if(not defined $init)
    {
        if(defined $self->init)
        {
            $init = $self->init;
        }
        else
        {
            $init = $default_init;
        }
    }
    if(not defined $self->shape or not @{ $self->shape } or product(@{ $self->shape }) <= 0)
    {
        if($self->allow_deferred_init)
        {
            $self->_deferred_init([$init, $ctx, $default_init, undef]);
            return;
        }
        confess("Cannot initialize Parameter '${\ $self->name }' because it has ".
                "invalid shape: @{$self->shape//[]}.");
    }
    $self->_deferred_init([$init, $ctx, $default_init, undef]);
    $self->_finish_deferred_init;
}

=head2 reset_ctx

        Re-assign Parameter to other contexts.

        :$ctx : AI::MXNet::Context or array ref of AI::MXNet::Context, default AI::MXNet::Context->current_ctx.
        Assign Parameter to given context. If ctx is a list of Context, a
        copy will be made for each context.
=cut

method reset_ctx(Maybe[AI::MXNet::Context|ArrayRef[AI::MXNet::Context]] :$ctx=AI::MXNet::Context->current_ctx)
{
    if(blessed($ctx) and $ctx->isa('AI::MXNet::Context'))
    {
        $ctx = [$ctx];
    }
    if(defined $self->_data)
    {
        my $data = $self->_reduce;
        AI::MXNet::AutoGrad->pause(sub {
            $self->_init_impl($data, $ctx);
        });
    }
    elsif(@{ $self->_deferred_init })
    {
        my ($init, undef, $default_init, $data) = @{ $self->_deferred_init };
        $self->_deferred_init([$init, $ctx, $default_init, $data]);
    }
    else
    {
        confess("Cannot reset context for Parameter '${ \ $self->name }' because it ".
                "has not been initialized.");
    }
}

=head2 set_data

    Sets this parameter's value on all contexts to data.
=cut

method set_data($data)
{
    $self->shape($data->shape);
    if(not defined $self->_data)
    {
        assert(
            (@{ $self->_deferred_init }),
            "Parameter '${\ $self->name }' has not been initialized"
        );
        $self->_deferred_init->[3] = $data;
        return;
    }

    # if update_on_kvstore, we need to make sure the copy stored in kvstore is in sync
    if($self->_trainer and $self->_trainer->_kv_initialized and $self->_trainer->update_on_kvstore)
    {
        if(!grep { Scalar::Util::refaddr($self) == Scalar::Util::refaddr($_) } @{ $self->_trainer->_params_to_init })
        {
            $self->_trainer->_reset_kvstore();
        }
    }
    for my $arr (@{ $self->_check_and_get($self->_data, []) })
    {
        $arr .= $data;
    }
}

=head2 row_sparse_data 

        Returns a copy of the 'row_sparse' parameter on the same context as row_id's.
        The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized on this context before.

        Parameters
        ----------
        $row_id: AI::MXNet::NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        AI::MXNet::NDArray on row_id's context
=cut

method row_sparse_data(AI::MXNet::NDArray $row_id)
{
    if($self->stype ne 'row_sparse')
    {
        confess(
            "Cannot return a copy of Parameter ${\ $self->name } via row_sparse_data() ".
            "because its storage type is ${\ $self->stype }. Please use data() instead."
        );
    }
    return $self->_get_row_sparse($self->_data, $row_id->context, $row_id);
}

=head2 list_row_sparse_data

        Returns copies of the 'row_sparse' parameter on all contexts, in the same order
        as creation. The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized before.

        Parameters
        ----------
        $row_id: AI::MXNet::NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        array ref of AI::MXNet::NDArrays
=cut

method list_row_sparse_data(AI::MXNet::NDArray $row_id)
{
    if($self->stype ne 'row_sparse')
    {
        confess(
            "Cannot return copies of Parameter '${\ $self->name }' on all contexts via ".
            "list_row_sparse_data() because its storage type is ${\ $self->stype }. Please ".
            "use data() instead."
        );
    }
    return $self->_get_row_sparse($self->_data, [], $row_id);
}

=head2 data

        Returns a copy of this parameter on one context. Must have been
        initialized on this context before. For sparse parameters, use
        row_sparse_data instead.

        Parameters
        ----------
        ctx : Context
            Desired context.

        Returns
        -------
        NDArray on ctx
=cut

method data(Maybe[AI::MXNet::Context] $ctx=)
{
    if($self->stype ne 'default')
    {
        $ctx //= AI::MXNet::Context->current_ctx;
        confess(
            "Cannot return a copy of Parameter '${\ $self->name }' on ctx $ctx via data() ".
            "because its storage type is ${\ $self->stype }. Please use row_sparse_data() ".
            "instead."
        );
    }
    return $self->_check_and_get($self->_data, $ctx);
}

=head2 list_data

        Returns copies of this parameter on all contexts, in the same order
        as creation. For sparse parameters, use list_row_sparse_data
        instead.
=cut

method list_data()
{
    if($self->stype ne 'default')
    {
        confess(
            "Cannot return a copies of Parameter '${\ $self->data }' on all contexts via list_data() ".
            "because its storage type is ${\ $self->stype }. Please use row_sparse_data() ".
            "instead."
        );
    }
    return $self->_check_and_get($self->_data, [])
}

=head2 grad

        Returns a gradient buffer for this parameter on one context.

        Parameters
        ----------
        ctx : Context
            Desired context.
=cut

method grad(Maybe [AI::MXNet::Context] $ctx=)
{
    if(defined $self->_data and not defined $self->_grad)
    {
        confess(
            "Cannot get gradient array for Parameter ${\ $self->name } ".
            "because grad_req='null'"
        );
    }
    return $self->_check_and_get($self->_grad, $ctx);
}

=head2 list_grad

        Returns gradient buffers on all contexts, in the same order
        as 'values'.
=cut

method list_grad()
{
    if(defined $self->_data and not defined $self->_grad)
    {
        confess(
            "Cannot get gradient array for Parameter ${\ $self->name } ".
            "because grad_req='null'"
        );
    }
    return $self->_check_and_get($self->_grad, []);
}

=head2 list_ctx

        Returns a list of contexts this parameter is initialized on.
=cut

method list_ctx()
{
    if(not defined $self->_data)
    {
        if(@{ $self->_deferred_init })
        {
            return $self->_deferred_init->[1];
        }
        confess("Parameter ${\ $self->name } has not been initialized");
    }
    return $self->_ctx_list;
}

=head2 zero_grad

        Sets gradient buffer on all contexts to 0. No action is taken if
        parameter is uninitialized or doesn't require gradient.
=cut

method zero_grad()
{
    return unless defined $self->_grad;
    AI::MXNet::NDArray->zeros_like($_, { out => $_ }) for @{ $self->_grad };
}

=head2 var

        Returns a symbol representing this parameter.
=cut

method var()
{
    if(not defined $self->_var)
    {
        $self->_var(
            AI::MXNet::Symbol->var(
                $self->name, shape => $self->shape, dtype => $self->dtype,
                lr_mult => $self->lr_mult, wd_mult => $self->wd_mult,
                init => $self->init, stype => $self->stype
            )
        );
    }
    return $self->_var;
}

=head2 cast

    Cast data and gradient of this Parameter to a new data type.

    Parameters
     ----------
    $dtype : Dtype
    The new data type.
=cut

method cast(Dtype $dtype)
{
    $self->dtype($dtype);
    return unless defined $self->_data;
    AI::MXNet::AutoGrad->pause(sub {
        $self->_data([map { $_->astype($dtype) } @{ $self->_data }]);
        return unless defined $self->_grad;
        $self->_grad([map { $_->astype($dtype) } @{ $self->_grad }]);
        AI::MXNet::AutoGrad->mark_variables($self->_data, $self->_grad, grad_reqs => $self->grad_req);
    });
}

__PACKAGE__->AI::MXNet::NS::register('AI::MXNet::Gluon');

package AI::MXNet::Gluon::Constant;
use strict;
use warnings;
use Mouse;
extends 'AI::MXNet::Gluon::Parameter';

=head1 NAME 

    AI::MXNet::Gluon::Constant - A constant parameter for holding immutable tensors.
=cut

=head1 DESCRIPTION

    A constant parameter for holding immutable tensors.
    Constants are ignored by autograd and Trainer, thus their values
    will not change during training. But you can still update their values
    manually with the set_data method.

    Constants can be created with either

        $const = mx->gluon->Constant('const', [[1,2],[3,4]]);

    or

        package Block;
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        sub BUILD
        {
            $self->const($self->params->get_constant('const', [[1,2],[3,4]]));
        }

    Constructor Attributes
    ----------
    name : str
        Name of the parameter.
    value : AcceptableInput (perl array, pdl, ndarray, etc)
        Initial value for the constant.
=cut

use Mouse;
use AI::MXNet::Base;
use Scalar::Util qw(refaddr);
around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments() { ['name', 'value'] }
has 'value'     => (is => 'rw', isa => 'AcceptableInput');
has '+_grad_req' => (is => 'rw', default => 'null');
use overload '""' => sub {
        my $self = shift;
        "Constant " . $self->name.
        " (shape=(" . join(', ', @{ $self->shape//[] }) .")".
        ", dtype=" . $self->dtype.
        ", stype=" . $self->stype.")"
    },
    fallback => 1;


sub BUILD
{
    my $self = shift;
    if(not (blessed $self->value and $self->value->isa('AI::MXNet::NDArray')))
    {
        $self->value(AI::MXNet::NDArray->array($self->value, dtype => $self->dtype));
    }
    $self->shape($self->value->shape);
    my $init = "AI::MXNet::Gluon::Constant::Init_${\ $self->name }_${\ refaddr($self) }";
    my $tmp =<<"EOP";
    package $init;
    use Mouse;
    extends 'AI::MXNet::Initializer';
    sub _init_weight
    {
        \$self->value->copyto(\$_[2]);
    }
    $init->register;
    1;
EOP
    eval $tmp;
    $self->init($init->new);
}

method grad_req($req=)
{
    if(defined $req and $req ne 'null')
    {
        AI::MXNet::Logging->warning(
            'Constant parameter "%s" does not support '.
            'grad_req other than "null", and new value "%s" '.
            'is ignored.',
            $self->name, $req
        );
    }
    return 'null';
}

package AI::MXNet::Gluon::ParameterDict;
use AI::MXNet::Base;
=head1 NAME

    AI::MXNet::Gluon::ParameterDict - A dictionary managing a set of parameters.
=cut

=head1 DESCRIPTION

    Parameters
    ----------
    prefix : str, default ''
        The prefix to be prepended to all Parameters' names created by this dict.
    shared : ParameterDict or undef
        If not undef, when this dict's `get` method creates a new parameter, will
        first try to retrieve it from `shared` dict. Usually used for sharing
        parameters with another `Block`.
=cut

use Mouse;
has _prefix => (is => 'ro', isa => 'Str', init_arg => 'prefix', default => '');
has _shared => (is => 'rw', isa => 'Maybe[AI::MXNet::Gluon::ParameterDict]', init_arg => 'shared');
has _params => (is => 'rw', init_arg => undef);

around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments() { [qw/prefix shared/] }

sub BUILD
{
    my $self = shift;
    $self->_params(Hash::Ordered->new);
}

use overload
    '""'   => sub {
        my $self = shift;
        my $name = $self->_prefix ? $self->_prefix." " : '';
        my $content = join("\n", map { AI::MXNet::Base::_indent("   $_", 2) } $self->values);
        return "$name(\n$content\n)";
    },
    '@{}'  => sub { my @tmp = shift->_params->as_list; \@tmp },
    fallback => 1;

method items()
{
    return @{$self};
}

method keys()
{
    return $self->_params->keys;
}

method values()
{
    return $self->_params->values;
}

method prefix()
{
    $self->_prefix;
}

method params()
{
    $self->_params;
}

method _get_impl($name)
{
    if($self->_params->exists($name))
    {
        return $self->_params->get($name);
    }
    if(defined $self->_shared and $self->_shared->_params->exists($name))
    {
        $self->_params->set($name => $self->_shared->_params->get($name));
        return $self->_params->get($name);
    }
    return undef;
}

=head2 get

        Retrieves a 'AI::MXNet::Gluon::Parameter' with name '$self->prefix.$name'. If not found,
        'get' will first try to retrieve it from 'shared' dict. If still not
        found, 'get' will create a new 'AI::MXNet::Gluon::Parameter' with key-word arguments and
        insert it to self.

        Parameters
        ----------
        name : str
            Name of the desired Parameter. It will be prepended with this dictionary's
            prefix.
        %kwargs : hash
            The rest of key-word arguments for the created `Parameter`.

        Returns
        -------
        Parameter
            The created or retrieved `Parameter`.
=cut

use Data::Dumper;
method get(Str $name, %kwargs)
{
    $name = $self->prefix . $name;
    my $param = $self->_get_impl($name);
    if(not defined $param)
    {
        $param = AI::MXNet::Gluon::Parameter->new($name, %kwargs);
        $self->_params->set($name => $param);
    }
    else
    {
        while(my ($k, $v) = each %kwargs)
        {
            if($param->can($k))
            {
                if(defined $param->$k)
                {
                    my $existing = $param->$k;
                    if($k eq 'shape' and @{$v} == @{$existing})
                    {
                        my @inferred_shape;
                        my $matched = 1;
                        for(zip($v, $existing))
                        {
                            my ($dim1, $dim2) = @$_;
                            if($dim1 != $dim2 and $dim1 * $dim2 != 0)
                            {
                                $matched = 0;
                                 last;
                            }
                            elsif($dim1 == $dim2)
                            {
                                push @inferred_shape, $dim1;
                            }
                            elsif($dim1 == 0)
                            {
                                push @inferred_shape, $dim2;
                            }
                            else
                            {
                                push @inferred_shape, $dim1;
                            }
                        }
                        if($matched)
                        {
                            $param->_shape(\@inferred_shape);
                            next;
                        }
                    }
                    elsif($k eq 'dtype' and ($v//'') eq ($existing//''))
                    {
                        next;
                    }
                    assert(
                        (not defined $v or Dumper($v) eq Dumper($param->$k)),
                        "Cannot retrieve Parameter $name because desired attribute ".
                        "does not match with stored for attribute $k: ".
                        "desired ".Dumper($v)." vs stored ". Dumper($param->$k)
                    );
                }
                else
                {
                    $param->$k($v);
                }
            }
            else
            {
                confess("unknown param $k, $v");
            }
        }
    }
    return $param;
}

=head2 update

    Copies all Parameters in $other to self.
=cut

method update($other, Maybe[Str] $select=)
{
    my @keys = $other->keys;
    for my $k (grep { not defined $select or /$select/ } @keys)
    {
        if($self->_params->exists($k))
        {
            assert(
                ($self->_params->get($k) eq $other->_params->get($k)),
                "Cannot update self with other because they have different ".
                "Parameters with the same name $k"
            );
        }
        else
        {
            $self->_params->set($k => $other->_params->get($k));
        }
    }
}

=head2 get_constant

        Retrieves AI::MXNet::Gluon::Constant with name $self->prefix.$name. If not found,
        'get' will first try to retrieve it from "shared" dictionary. If still not
        found, 'get' will create a new Constant with key-word
        arguments and insert it to self.

        Parameters
        ----------
        name : str
            Name of the desired Constant. It will be prepended with this dictionary's
            prefix.
        value : array-like
            Initial value of constant.

        Returns
        -------
        Constant
            The created or retrieved Constant.
=cut

method get_constant(Str $name, Maybe[AcceptableInput] $value=)
{
    $name = $self->prefix . $name;
    my $param = $self->_get_impl($name);
    if(not defined $param)
    {
        if(not defined $value)
        {
            confess(
                "No constant named '$name'. Please specify value ".
                "if you want to create a new constant."
            );
        }
        $param = AI::MXNet::Gluon::Constant->new($name, $value);
        $self->_params->set($name, $param);
    }
    elsif(defined $value)
    {
        confess("reinit of Constant $name is not allowed");
    }
    return $param;
}

=head2 initialize

        Initializes all Parameters managed by this dictionary to be used for 'NDArray'
        API. It has no effect when using 'Symbol' API.

        Parameters
        ----------
        :$init : Initializer
            Global default Initializer to be used when AI::MXNet::Gluon::Parameter->init is undef.
            Otherwise, AI::MXNet::Gluon::Parameter->init takes precedence.
        :$ctx : AI::MXNet::Context or array ref of AI::MXNet::Context objects
            Keeps a copy of Parameters on one or many context(s).
        :$force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        :$verbose : bool, default False
            Whether to force re-initialization if parameter is already initialized.
=cut

method initialize(
    Initializer                                            :$init=AI::MXNet::Initializer->Uniform(),
    Maybe[AI::MXNet::Context|ArrayRef[AI::MXNet::Context]] :$ctx=,
    Bool                                                   :$verbose=0,
    Bool                                                   :$force_reinit=0
)
{
    if($verbose)
    {
        $init->set_verbosity(verbose=>$verbose);
    }
    $_->initialize(ctx => $ctx, default_init => $init, force_reinit => $force_reinit) for $self->values;
}

=head2 zero_grad

    Sets all Parameters' gradient buffer to 0.
=cut

method zero_grad()
{
    $_->zero_grad for $self->values;
}

=head2 reset_ctx

    Re-assign all Parameters to other contexts.

    $ctx : AI::MXNet::Context or array ref of AI::MXNet::Context objects, defaults to AI::MXNet::Context->current_ctx().
            Assign Parameter to given context. If $ctx is an array ref of AI::MXNet::Context objects, a
            copy will be made for each context.
=cut

method reset_ctx(AI::MXNet::Context|ArrayRef[AI::MXNet::Conetxt] $ctx=AI::MXNet::Context->current_ctx)
{
    $_->reset_ctx($ctx) for $self->values;
}

=head2 setattr

        Set an attribute to a new value for all Parameters.

        For example, set grad_req to null if you don't need gradient w.r.t a
        model's Parameters::

            $model->collect_params()->setattr(grad_req => 'null');

        or change the learning rate multiplier::

            $model->collect_params()->setattr(lr_mult => 0.5);

        Parameters
        ----------
        $name : str
            Name of the attribute.
        $value : valid type for attribute name
            The new value for the attribute.
=cut

method setattr($name, $value)
{
    $_->$name($value) for $self->values;
}


=head2 save

    Save parameters to file.

    $filename : str
        Path to parameter file.
    $strip_prefix : str, default ''
    Strip prefix from parameter names before saving.
=cut

method save(Str $filename, Str $strip_prefix='')
{
    my %arg_dict = ();
    for my $param ($self->values())
    {
        my $weight = $param->_reduce();
        if(not $param->name =~ /^$strip_prefix/)
        {
            confess(
                "Prefix $strip_prefix is to be striped before saving, but Parameter ".
                "${\ $param->name } does not start with $strip_prefix. If you are using Block.save_params, ".
                "This may be due to your Block shares parameters from other ".
                "Blocks or you forgot to use `with name_scope()`` during init. ".
                "Consider switching to Block.collect_params.save and ".
                "Block.collect_params.load instead."
            );
        }
        $arg_dict{ substr($param->name, length $strip_prefix) } = $weight;
    }
    AI::MXNet::NDArray->save($filename, \%arg_dict);
}

=head2

        Load parameters from file.

        $filename : str
            Path to parameter file.
        :$ctx : AI::MXNet::Context or array ref of AI::MXNet::Context objects
            Context(s) initialize loaded parameters on.
        :$allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        :$ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this ParameterDict.
        :$restore_prefix : str, default ''
            prepend prefix to names of stored parameters before loading.
=cut

method load(
    Str                                              $filename,
    AI::MXNet::Context|ArrayRef[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Bool                                            :$allow_missing=0,
    Bool                                            :$ignore_extra=0,
    Str                                             :$restore_prefix=''
)
{
    if($restore_prefix)
    {
        for my $name ($self->keys())
        {
            assert(
                ($name =~ /^$restore_prefix/),
                "restore_prefix is $restore_prefix but Parameters name $name does not start ".
                "with $restore_prefix"
            );
        }
    }
    my $lprefix  = length $restore_prefix;
    my %orig_load = %{ AI::MXNet::NDArray->load($filename) };
    my %arg_dict  = map { my $k = $_; s/^(?:arg|aux)://; ($restore_prefix.$_, $orig_load{$k}) } keys %orig_load;
    if(not $allow_missing)
    {
        for my $name ($self->keys())
        {
            assert(
                (exists $arg_dict{ $name }),
                sprintf("Parameter %s is missing in file %s", substr($name, $lprefix), $filename)
            );
        }
    }
    for my $name (keys %arg_dict)
    {
        if(not $self->_params->exists($name))
        {
            assert(
                $ignore_extra,
                sprintf(
                    "Parameter %s loaded from file %s is not present in ParameterDict",
                    substr($name, $lprefix),
                    $filename
                )
            );
            next;
        }
        $self->_params->get($name)->_load_init($arg_dict{$name}, $ctx);
    }
}

__PACKAGE__->AI::MXNet::NS::register('AI::MXNet::Gluon');

1;
