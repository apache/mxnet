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
package AI::MXNet::Gluon::Trainer;
use AI::MXNet::NS;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use IO::File;
use Mouse;


=head1 NAME

    AI::MXNet::Gluon::Trainer
=cut

=head1 DESCRIPTION

    Applies an `Optimizer` on a set of Parameters. Trainer should
    be used together with `autograd`.

    Parameters
    ----------
    params : AI::MXNet::Gluon::ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <https://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : hash ref
        Key-word arguments to be passed to optimizer constructor. For example,
        {learning_rate => 0.1}. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training. See help on
        mx->kvstore->create for more information.
    compression_params : hash ref
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {type => '2bit', threshold => 0.5}
        See AI::MXNet::KVStore->set_gradient_compression method for more details on gradient compression.
    update_on_kvstore : Bool, default undef
        Whether to perform parameter updates on kvstore. If undef, then trainer will choose the more
        suitable option depending on the type of kvstore.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer->learning_rate.
=cut

has 'params'             => (is => 'rw', isa => 'HashRef|ArrayRef|AI::MXNet::Gluon::ParameterDict');
has 'optimizer'          => (is => 'ro', isa => 'Optimizer');
has 'optimizer_params'   => (is => 'ro', isa => 'Maybe[HashRef]');
has 'compression_params' => (is => 'ro', isa => 'Maybe[HashRef]');
has 'kvstore'            => (is => 'rw', isa => 'Maybe[KVStore]', default => 'device');
has 'update_on_kvstore'  => (is => 'rw', isa => 'Maybe[Bool]');
has [qw/_scale _contexts
    _kv_initialized
    _param2idx
    _kvstore_params
    _contains_sparse
    _params_to_init
    _updaters
    _optimizer/]       => (is => 'rw', init_arg => undef);
around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments()
{
    [qw/params optimizer optimizer_params kvstore compression_params update_on_kvstore/]
}

sub BUILD
{
    my $self = shift;
    my @params;
    if(blessed $self->params)
    {
        @params = $self->params->values;
    }
    elsif(ref $self->params eq 'HASH')
    {
        @params = values %{ $self->params };
    }
    else
    {
        @params = @{ $self->params };
    }
    $self->params([]);
    $self->_contains_sparse(0);
    $self->_param2idx({});
    for(enumerate(\@params))
    {
        my ($i, $param) = @$_;
        if(not(blessed $param and $param->isa('AI::MXNet::Gluon::Parameter')))
        {
            confess(
                "First argument must be a array or hash of Parameters, ".
                "got list of [$param]."
            );
        }
        $self->_param2idx->{ $param->name } = $i;
        push @{ $self->params }, $param;
        $param->_set_trainer($self);
        if($param->stype ne 'default')
        {
            $self->_contains_sparse(1);
        }
    }
    my $optimizer_params = $self->optimizer_params//{};
    $self->_scale(delete $optimizer_params->{rescale_grad}//1);
    $self->_contexts($self->_check_contexts);
    $self->_init_optimizer($self->optimizer, $optimizer_params);
    $self->_kvstore_params({
        kvstore => $self->kvstore,
        update_on_kvstore => $self->update_on_kvstore
    });
    $self->_kv_initialized(0);
    $self->kvstore(undef);
    $self->update_on_kvstore(undef);
    $self->_params_to_init([]);
    $self->_reset_kvstore();
}

method _check_contexts()
{
    my $contexts;
    for my $param (@{ $self->params })
    {
        my $ctx = $param->list_ctx;
        assert(
            (not defined $contexts or join('', @{ $contexts }) eq join('', @{ $ctx })),
            "All Parameters must be initialized on the same set of contexts, ".
            "but Parameter ${\ $param->name } is initialized on @{ $ctx//[] } while previous Parameters ".
            "are initialized on @{ $contexts//[] }."
        );
        $contexts = $ctx;
    }
    return $contexts;
}

method _init_optimizer($optimizer, $optimizer_params)
{
    my %param_dict = map { $_ => $self->params->[$_] } 0 .. @{ $self->params } - 1;
    if(blessed $optimizer and $optimizer->isa('AI::MXNet::Optimizer'))
    {
        assert(
            (not %{ $optimizer_params }),
            "optimizer_params must be empty if optimizer is an instance of ".
            "Optimizer instead of str"
        );
        $self->_optimizer($optimizer);
        $self->_optimizer->param_dict(\%param_dict);
    }
    else
    {
        $self->_optimizer(
            AI::MXNet::Optimizer->create(
                $optimizer, param_dict => \%param_dict,
                %{ $optimizer_params }
            )
        );
    }
    $self->_updaters([
        map { AI::MXNet::Optimizer->get_updater($self->_optimizer) } @{ $self->_contexts }
    ]);
}

method _init_params()
{
    assert(
        $self->_kv_initialized,
        "Cannot initialize parameters in KVStore ".
        "when KVStore is not initialized."
    );
    my @params_to_init;
    if($self->kvstore)
    {
        for my $param (@{ $self->_params_to_init })
        {
            if(@{ $param->_deferred_init })
            {
                push @params_to_init, $param;
            }
            else
            {
                my $param_arrays = $param->_check_and_get($param->_data, []);
                my $idx = $self->_param2idx->{ $param->name };
                $self->kvstore->init($idx, $param_arrays->[0]);
                if($param->stype eq 'default')
                {
                    $self->kvstore->pull($idx, out => $param_arrays, priority=>-$idx);
                }
            }
        }
    }
    $self->_params_to_init(\@params_to_init);
}

method _reset_kvstore()
{
    if($self->kvstore and $self->kvstore->type =~ /dist/)
    {
        confess("Cannot reset distributed KVStore.");
    }
    $self->_kv_initialized(0);
    $self->kvstore(undef);
    $self->update_on_kvstore(undef);
    $self->_params_to_init([@{ $self->params }]);
}

method _init_kvstore()
{
    my $config = $self->_kvstore_params;
    my ($kvstore, $update_on_kvstore);
    if($self->_contains_sparse)
    {
        ($kvstore, $update_on_kvstore) = AI::MXNet::Module::_create_sparse_kvstore($config->{kvstore});
        # update_on_kvstore is set to False by the user
        if(defined $config->{update_on_kvstore} and not $config->{update_on_kvstore})
        {
            confess(
                "Cannot set update_on_kvstore to False when sparse ".
                "gradients and/or sparse weights are present."
            )
        }
    }
    else
    {
        my %arg_arrays = map { $_->name => $_->data($self->_contexts->[0]) } @{ $self->params };
        ($kvstore, $update_on_kvstore) = AI::MXNet::Module::_create_kvstore(
            $config->{kvstore}, scalar(@{$self->_contexts }), \%arg_arrays
        );
        if(defined $config->{update_on_kvstore})
        {
            $update_on_kvstore = $config->{update_on_kvstore};
        }
    }
    if($kvstore)
    {
        if($self->compression_params)
        {
            $kvstore->set_gradient_compression($self->compression_params);
        }
        # kv->pull(row_sparse_grad) is not supported
        if($kvstore->type =~ /dist/ and not $self->_contains_sparse)
        {
            $update_on_kvstore = 0;
        }
        if($update_on_kvstore)
        {
            # optimizer preferably needs to be set before init for multiprecision
            $kvstore->set_optimizer($self->_optimizer);
        }
        $self->kvstore($kvstore);
        $self->update_on_kvstore($update_on_kvstore);
    }
    else
    {
        $self->kvstore(undef);
        $self->update_on_kvstore(undef);
    }
    $self->_kv_initialized(1);
}

# Internal method to invoke pull operations on KVStore. If $full_idx is set to 1,
# $kv->pull is preferred instead of $kv->row_sparse_pull.

method _row_sparse_pull($parameter, $out, $row_id, $full_idx=0)
{
    # initialize kv and params if not already
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });
    my $idx = $self->_param2idx->{ $parameter->name };
    if($full_idx and not $self->kvstore->type =~ /dist/)
    {
        assert($row_id->size == $out->shape->[0]);
        $self->kvstore->pull($idx, out => $out, priority => -$idx, ignore_sparse => 0);
    }
    else
    {
        $self->kvstore->row_sparse_pull($idx, out => $out, row_ids => $row_id, priority => -$idx);
    }
}

=head2 step

        Makes one step of parameter update. Should be called after
        `autograd->backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        $batch_size : Int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        $ignore_stale_grad : Bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
=cut

method step(Int $batch_size, Bool $ignore_stale_grad=0)
{
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });
    $self->_optimizer->rescale_grad($self->_scale/$batch_size);
    $self->_allreduce_grads();
    $self->_update($ignore_stale_grad);
}

=head2 allreduce_grads

        For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
=cut

method allreduce_grads()
{
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });
    assert(
        (not ($self->kvstore and $self->update_on_kvstore)),
        'allreduce_grads() when parameters are updated on kvstore '.
        'is not supported. Try setting `update_on_kvstore` '.
        'to False when creating trainer.'
    );
    $self->_allreduce_grads();
}

method _allreduce_grads()
{
    if($self->kvstore)
    {
        for(enumerate($self->params))
        {
            my ($i, $param) = @$_;
            if($param->grad_req ne 'null')
            {
                $self->kvstore->push($i, $param->list_grad(), priority=>-$i);
                if(not $self->update_on_kvstore)
                {
                    $self->kvstore->pull($i, out => $param->list_grad(), priority=>-$i);
                }
            }
        }
    }
}

method learning_rate(Maybe [Num] $lr)
{
    if(not blessed $self->_optimizer)
    {
        AI::MXNet::Logging->warning(
            "Optimizer has to be defined before its learning ".
            "rate can be accessed."
        );
        return;
    }
    else
    {
        if(defined $lr)
        {
            $self->_optimizer->lr($lr);
        }
        return $self->_optimizer->lr;
    }
}

=head2 set_learning_rate

        Sets a new learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            The new learning rate of the optimizer.
=cut

method set_learning_rate(Num $lr)
{
    $self->learning_rate($lr);
}

=head2 update

        Makes one step of parameter update.

        Should be called after autograd->backward() and outside of record() scope,
        and after trainer->update`.


        For normal parameter updates, step() should be used, which internally calls
        allreduce_grads() and then update(). However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call allreduce_grads() and update() separately.

        Parameters
        ----------
        $batch_size : Int
            Batch size of data processed. Gradient will be normalized by `1/$batch_size`.
            Set this to 1 if you normalized loss manually with $loss = mean($loss).
        $ignore_stale_grad : Bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by backward() after last step) and skip update.
=cut

method update(Int $batch_size, Bool $ignore_stale_grad=0)
{
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });
    assert(
        (not ($self->kvstore and $self->update_on_kvstore)),
        'update() when parameters are updated on kvstore '.
        'is not supported. Try setting `update_on_kvstore` '.
        'to False when creating trainer.'
    );
    $self->_optimizer->rescale_grad($self->_scale/$batch_size);
    $self->_update($ignore_stale_grad);
}

method _update(Bool $ignore_stale_grad=0):
{
    for(enumerate($self->params))
    {
        my ($i, $param) = @$_;
        next if($param->grad_req eq 'null');

        if(not $ignore_stale_grad)
        {
            for my $data (@{ $param->_check_and_get($param->_data, []) })
            {
                if(not $data->_fresh_grad)
                {
                    AI::MXNet::Logging->warning(
                        "Gradient of Parameter '%s' on context %s has not been updated ".
                        "by backward since last `step`. This could mean a bug in your ".
                        "model that made it only use a subset of the Parameters (Blocks) ".
                        "for this iteration. If you are intentionally only using a subset, ".
                        "call step with ignore_stale_grad=True to suppress this ".
                        "warning and skip updating of Parameters with stale gradient",
                        $param->name, $data->context
                    );
                }
            }
        }
        if($self->kvstore and $self->update_on_kvstore)
        {
            if($param->stype eq 'default')
            {
                # 'row_sparse' parameters are not pulled immediately - they're pulled
                # in `SparseBlock.sparse_forward`
                $self->kvstore->pull($i, out => $param->list_data(), priority=>-$i);
            }
            next;
        }

        for(zip($self->_updaters, $param->list_data(), $param->list_grad()))
        {
            my ($upd, $arr, $grad) = @$_;
            if(not $ignore_stale_grad or $arr->_fresh_grad)
            {
                $upd->($i, $grad, $arr);
                $arr->_fresh_grad(0);
            }
        }
    }
}

=head2 save_states

        Saves trainer states (e.g. optimizer, momentum) to a file.

        Parameters
        ----------
        fname : str
            Path to output states file.
=cut

method save_states(Str $fname)
{
    assert(defined $self->_optimizer);
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });

    if($self->update_on_kvstore)
    {
        $self->kvstore->save_optimizer_states($fname, dump_optimizer=>1);
    }
    else
    {
        open(F, ">$fname") or Carp::confess("can not open $fname: $1");
        print F $self->_updaters->[0]->get_states(dump_optimizer => 1);
        close(F);
    }
}

=head2 load_states

        Loads trainer states (e.g. optimizer, momentum) from a file.

        Parameters
        ----------
        fname : str
            Path to input states file.
=cut

method load_states(Str $fname)
{
    $self->_init_kvstore() unless $self->_kv_initialized;
    $self->_init_params() if scalar(@{ $self->_params_to_init });

    if($self->update_on_kvstore)
    {
        $self->kvstore->load_optimizer_states($fname);
        $self->_optimizer($self->kvstore->_updater->optimizer);
        $self->_optimizer->param_dict({ map { $_->[0] => $_->[1] } enumerate($self->params) });
    }
    else
    {
        my $states = join('', IO::File->new($fname)->getlines);
        for my $updater (@{ $self->_updaters })
        {
            $updater->set_states($states);
            $updater->optimizer($self->_updaters->[0]->optimizer);
        }
        $self->_optimizer($self->_updaters->[0]->optimizer);
    }
}

__PACKAGE__->AI::MXNet::NS::register('AI::MXNet::Gluon');

1;
