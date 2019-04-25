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

package AI::MXNet::Executor;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Context;
use Mouse;
use AI::MXNet::Types;
use AI::MXNet::Function::Parameters;

has 'handle'            => (is => 'ro', isa => 'ExecutorHandle', required => 1);
has 'arg_arrays'        => (is => 'rw', isa => 'Maybe[ArrayRef[AI::MXNet::NDArray]]');
has 'grad_arrays'       => (is => 'rw', isa => 'Maybe[ArrayRef[Undef|AI::MXNet::NDArray]]');
has 'aux_arrays'        => (is => 'rw', isa => 'Maybe[ArrayRef[AI::MXNet::NDArray]]');
has '_symbol'           => (is => 'rw', init_arg => 'symbol',    isa => 'AI::MXNet::Symbol');
has '_ctx'              => (is => 'rw', init_arg => 'ctx',       isa => 'AI::MXNet::Context' );
has '_grad_req'         => (is => 'rw', init_arg => 'grad_req',  isa => 'Maybe[Str|ArrayRef[Str]|HashRef[Str]]');
has '_group2ctx'        => (is => 'rw', init_arg => 'group2ctx', isa => 'Maybe[HashRef[AI::MXNet::Context]]');
has [qw/_arg_dict
        _grad_dict
        _aux_dict
        _output_dict
        outputs
    /]                  => (is => 'rw', init_arg => undef);
=head1 NAME

    AI::MXNet::Executor - The actual executing object of MXNet.
=cut

=head1 SYNOPSIS

    my $executor = $sym->bind(
        ctx       => mx->Context('cpu'),
        args      => [$lhs_arr, $rhs_arr],
        args_grad => [$lhs_grad, $rhs_grad]
    );
    $executor->forward(1);
    print $executor->outputs->[0]->aspdl;
=cut

=head2 new

    Constructor, used by AI::MXNet::Symbol->bind and by AI::MXNet::Symbol->simple_bind.

    Parameters
    ----------
    handle: ExecutorHandle
        ExecutorHandle is generated by calling bind.

    See Also
    --------
    AI::MXNet::Symbol->bind : how to create the AI::MXNet::Executor.
=cut

sub BUILD
{
    my $self = shift;
    my ($symbol, $ctx, $grad_req, $group2ctx)
        =
    ($self->_symbol, $self->_ctx, $self->_grad_req, $self->_group2ctx);
    $symbol = $symbol->deepcopy;
    $ctx    = $ctx->deepcopy;
    if(ref $grad_req)
    {
        if(ref $grad_req eq 'ARRAY')
        {
            $grad_req = [ @{ $grad_req }];
        }
        elsif(ref $grad_req eq 'HASH')
        {
            $grad_req = { %{ $grad_req } };

        }
    }
    if(ref $group2ctx)
    {
        $group2ctx = { %{ $group2ctx } };
    }
    $self->_symbol($symbol);
    $self->_ctx($ctx);
    $self->_grad_req($grad_req);
    $self->_group2ctx($group2ctx);
    $self->outputs($self->_get_outputs);
}

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::ExecutorFree(shift->handle));
}

# Get the dictionary given name and ndarray pairs.
func _get_dict(
    ArrayRef[Str]                       $names,
    ArrayRef[Maybe[AI::MXNet::NDArray]] $ndarrays
)
{
    my %nset = ();
    for my $nm (@{ $names })
    {
        if(exists $nset{ $nm })
        {
            confess("Duplicate names detected, @$names")
        }
        $nset{ $nm }++;
    }
    my %ret;
    @ret{ @{ $names } } = @{ $ndarrays };
    return \%ret;
}

=head2 outputs

    The output ndarrays bound to this executor.

    Returns
    -------
    An array ref with AI::MXNet::NDArray objects bound to the heads of the executor.
=cut

method _get_outputs()
{
    return [
            map {
                AI::MXNet::NDArray->_ndarray_cls($_)
            }
            @{ check_call(AI::MXNetCAPI::ExecutorOutputs($self->handle)) }
    ];
}

=head2 forward

    Calculate the outputs specified by the bound symbol.

    Parameters
    ----------
    $is_train=0: Bool, optional
        whether this forward is for evaluation purpose. If True,
        a backward call is expected to follow. Otherwise following
        backward is invalid.

    %kwargs
        Additional specification of input arguments.

    Examples
    --------
        >>> # doing forward by specifying data
        >>> $texec->forward(1, data => $mydata);
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> $mydata->copyto($texec->arg_dict->{'data'});
        >>> $texec->forward(1);
        >>> # doing forward by specifying data and get outputs
        >>> my $outputs = $texec->forward(1, data => $mydata);
        >>> print $outputs->[0]->aspdl;
=cut

method forward(Int $is_train=0, %kwargs)
{
    if(%kwargs)
    {
        my $arg_dict = $self->arg_dict;
        while (my ($name, $array) = each %kwargs)
        {
            if(not find_type_constraint('AcceptableInput')->check($array))
            {
                confess('only accept keyword argument of NDArrays/PDLs/Perl Array refs');
            }
            if(not exists $arg_dict->{ $name })
            {
                confess("unknown argument $name");
            }
            if(not blessed($array) or not $array->isa('AI::MXNet::NDArray'))
            {
                $array = AI::MXNet::NDArray->array($array);
            }
            if(join(',', @{ $arg_dict->{$name}->shape }) ne join(',', @{ $array->shape }))
            {
                my $expected = $arg_dict->{$name}->shape;
                my $got = $array->shape;
                confess("Shape not match! Argument $name, need: @$expected, received: @$got'");
            }
            $arg_dict->{ $name } .= $array;
        }
    }
    check_call(AI::MXNetCAPI::ExecutorForward(
            $self->handle,
            $is_train
        )
    );
    return $self->outputs;
}

=head2 backward

    Do a backward pass to get the gradient of the arguments.

    Parameters
    ----------
    $out_grads : NDArray or an array ref of NDArrays or hash ref of NDArrays, optional.
        The gradient on the outputs to be propagated back.
        This parameter is only needed when bind is called
        on outputs that are not a loss function.

    $is_train : Bool, default 1
        Whether this backward is for training or inference. Note that in rare
        cases you want to call backward with is_train=0 to get gradient
        during inference.
=cut

method backward(
    Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|HashRef[AI::MXNet::NDArray]] $out_grads=,
    Bool $is_train=1
)
{
    $out_grads //= [];
    if(blessed $out_grads)
    {
        $out_grads = [$out_grads];
    }
    elsif(ref $out_grads eq 'HASH')
    {
        $out_grads = [ @{ $out_grads }{ @{ $self->symbol->list_outputs() } } ];
    }
    check_call(
        AI::MXNetCAPI::ExecutorBackwardEx(
            $self->handle,
            scalar(@{ $out_grads }),
            [map { $_->handle } @{ $out_grads }],
            $is_train
        )
    );
}

=head2 set_monitor_callback

    Install callback.

    Parameters
    ----------
    $callback : CodeRef
        Takes a string and an NDArrayHandle.
=cut

method set_monitor_callback(CodeRef $callback)
{
    check_call(
        AI::MXNetCAPI::ExecutorSetMonitorCallback(
            $self->handle,
            $callback
        )
    );
}

=head2 arg_dict

    Get a hash ref representation of the argument arrays.

    Returns
    -------
    $arg_dict : HashRef[AI::MXNet::NDArray]
        The map that maps a name of the arguments to the NDArrays.
=cut

method arg_dict()
{
    if(not defined $self->_arg_dict)
    {
        $self->_arg_dict(_get_dict(
                $self->_symbol->list_arguments(),
                $self->arg_arrays
            )
        );
    }
    return $self->_arg_dict;
}

=head2 grad_dict

    Get a hash ref representation of the gradient arrays.

    Returns
    -------
    $grad_dict : HashRef[AI::MXNet::NDArray]
        The map that maps a name of the arguments to the gradient NDArrays.
=cut

method grad_dict()
{
    if(not defined $self->_grad_dict)
    {
        $self->_grad_dict(_get_dict(
                $self->_symbol->list_arguments(),
                $self->grad_arrays
            )
        );
    }
    return $self->_grad_dict;
}

=head2 aux_dict

    Get a hash ref representation of the auxiliary states arrays.

    Returns
    -------
    $aux_dict : HashRef[AI::MXNet::NDArray]
        The map that maps a name of the auxiliary states to the NDArrays.
=cut

method aux_dict()
{
    if(not defined $self->_aux_dict)
    {
        $self->_aux_dict(_get_dict(
                $self->_symbol->list_auxiliary_states(),
                $self->aux_arrays()
            )
        );
    }
    return $self->_aux_dict;
}

=head2 output_dict

    Get a hash ref representation of the output arrays.

    Returns
    -------
    $output_dict : HashRef[AI::MXNet::NDArray]
        The map that maps a name of the outputs to the NDArrays.
=cut

method output_dict()
{
    if(not defined $self->_output_dict)
    {
        $self->_output_dict(_get_dict(
                $self->_symbol->list_outputs(),
                $self->outputs
            )
        );
    }
    return $self->_output_dict;
}

=head2 copy_params_from

    Copy parameters from arg_params, aux_params into the executor's internal array.

    Parameters
    ----------
    $arg_params : HashRef[AI::MXNet::NDArray]
        Parameters, hash ref of name to NDArray of arguments

    $aux_params= : Maybe[HashRef[AI::MXNet::NDArray]], optional
        Parameters, hash ref of name to NDArray of auxiliary states.

    $allow_extra_params= : Bool, optional
        Whether to allow extra parameters that are not needed by symbol
        If this is True, no error will be thrown when arg_params or aux_params
        contain extra parameters that is not needed by the executor.
=cut

method copy_params_from(
    HashRef[AI::MXNet::NDArray]        $arg_params,
    Maybe[HashRef[AI::MXNet::NDArray]] $aux_params=,
    Maybe[Bool]                        $allow_extra_params=
)
{
    my %arg_dict = %{ $self->arg_dict };
    while (my ($name, $array) = each %{ $arg_params })
    {
        if(exists $arg_dict{ $name })
        {
            my $dst = $arg_dict{ $name };
            $array->astype($dst->dtype)->copyto($dst);
        }
        elsif(not $allow_extra_params)
        {
            confess("Found name \"$name\" that is not in the arguments");
        }
    }
    if(defined $aux_params)
    {
        my %aux_dict = %{ $self->aux_dict };
        while (my ($name, $array) = each %{ $aux_params })
        {
            if(exists $aux_dict{ $name })
            {
                my $dst = $aux_dict{ $name };
                $array->astype($dst->dtype)->copyto($dst);
            }
            elsif(not $allow_extra_params)
            {
                confess("Found name \"$name\" that is not in the arguments");
            }
        }
    }
}

=head2 reshape

    Returns new executor with the same symbol and shared memory,
    but different input/output shapes.
    For runtime reshaping, variable length sequences, etc.
    The returned executor shares state with the current one,
    and cannot be used in parallel with it.

    Parameters
    ----------
    $kwargs : HashRef[Shape]
        new shape for arguments.
    :$partial_shaping : Bool
        Whether to allow changing the shape of unspecified arguments.
    :$allow_up_sizing : Bool
        Whether to allow allocating new ndarrays that's larger than the original.

    Returns
    -------
    $exec : AI::MXNet::Executor
        A new executor that shares memory with self.
=cut


method reshape(HashRef[Shape] $kwargs, Int :$partial_shaping=0, Int :$allow_up_sizing=0)
{
    my @provided_arg_shape_data;
    # argument shape index in sdata,
    # e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
    my @provided_arg_shape_idx = (0);
    my @provided_arg_shape_names = ();  # provided argument names
    while(my ($k, $v) = each %{ $kwargs })
    {
        if(ref $v eq 'ARRAY')
        {
            push @provided_arg_shape_names, $k;
            push @provided_arg_shape_data, @{ $v };
            push @provided_arg_shape_idx, scalar(@provided_arg_shape_data);
        }
    }

    my @ctx_map_keys;
    my @ctx_map_dev_types;
    my @ctx_map_dev_ids;

    if(ref $self->_group2ctx eq 'HASH')
    {
        while(my ($k, $v) = each %{ $self->_group2ctx })
        {
            push @ctx_map_keys, $k;
            push @ctx_map_dev_types, $v->device_type_id;
            push @ctx_map_dev_ids, $v->device_id;
        }
    }

    my $shared_handle = $self->handle;

    my ($in_args_and_grad_handles, $aux_state_handles, $handle) = check_call(
        AI::MXNetCAPI::ExecutorReshapeEx(
            $partial_shaping,
            $allow_up_sizing,
            $self->_ctx->device_type_id,
            $self->_ctx->device_id,
            scalar(@ctx_map_keys),
            \@ctx_map_keys,
            \@ctx_map_dev_types,
            \@ctx_map_dev_ids,
            scalar(@provided_arg_shape_names),
            \@provided_arg_shape_names,
            \@provided_arg_shape_data,
            \@provided_arg_shape_idx,
            $shared_handle
        )
    );
    my ($in_args_handles, $arg_grad_handles) = @{ $in_args_and_grad_handles };
    my @arg_arrays  = map { AI::MXNet::NDArray->_ndarray_cls($_) } @{ $in_args_handles };
    my @grad_arrays = map { defined($_) ? AI::MXNet::NDArray->_ndarray_cls($_) : undef } @{ $arg_grad_handles };
    my @aux_arrays  = map { AI::MXNet::NDArray->_ndarray_cls($_) } @{ $aux_state_handles };

    my $executor = __PACKAGE__->new(
        handle     => $handle,
        symbol    => $self->_symbol,
        ctx       => $self->_ctx,
        grad_req  => $self->_grad_req,
        group2ctx => $self->_group2ctx
    );
    $executor->arg_arrays(\@arg_arrays);
    $executor->grad_arrays(\@grad_arrays);
    $executor->aux_arrays(\@aux_arrays);
    return $executor;
}

=head2 debug_str

    A debug string about the internal execution plan.

    Returns
    -------
    $debug_str : Str
        Debug string of the executor.
=cut

method debug_str()
{
    return scalar(check_call(AI::MXNetCAPI::ExecutorPrint($self->handle)));
}

1;
