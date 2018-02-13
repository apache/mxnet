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
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training. See help on
        :any:`mxnet.kvstore.create` for more information.
=cut

has '_params'          => (is => 'rw', init_arg => 'params', isa => 'HashRef|ArrayRef|AI::MXNet::Gluon::ParameterDict');
has 'optimizer'        => (is => 'ro', isa => 'Optimizer');
has 'optimizer_params' => (is => 'ro', isa => 'Maybe[HashRef]');
has '_kv_store'        => (is => 'rw', init_arg => 'kvstore', isa => 'Maybe[KVStore]', default => 'device');
has [qw/_scale _contexts
    _kv_initialized
    _update_on_kvstore
    _updaters
    _optimizer/]       => (is => 'rw', init_arg => undef);
around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments() { ['params', 'optimizer', 'optimizer_params'] }

sub BUILD
{
    my $self = shift;
    my @params;
    if(blessed $self->_params)
    {
        @params = $self->_params->values;
    }
    elsif(ref $self->_params eq 'HASH')
    {
        @params = values %{ $self->_params };
    }
    else
    {
        @params = @{ $self->_params };
    }
    $self->_params([]);
    for my $param (@params)
    {
        if(not(blessed $param and $param->isa('AI::MXNet::Gluon::Parameter')))
        {
            confess(
                "First argument must be a array or hash of Parameters, ".
                "got list of [$param]."
            );
        }
        push @{ $self->_params }, $param;
    }
    my $optimizer_params = $self->optimizer_params//{};
    $self->_scale(delete $optimizer_params->{rescale_grad}//1);
    $self->_contexts($self->_check_contexts);
    $self->_init_optimizer($self->optimizer, $optimizer_params);
    $self->_kv_initialized(0);
}

method _check_contexts()
{
    my $contexts;
    for my $param (@{ $self->_params })
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
    my %param_dict = map { $_ => $self->_params->[$_] } 0 .. @{ $self->_params } - 1;
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

method _init_kvstore()
{
    my %arg_arrays = map { $_->name => $_->data($self->_contexts->[0]) } @{ $self->_params };
    my ($kvstore, $update_on_kvstore) = AI::MXNet::Module::_create_kvstore(
        $self->_kv_store, scalar(@{$self->_contexts }), \%arg_arrays
    );
    if($kvstore)
    {
        if($kvstore->type =~ /dist/)
        {
            $update_on_kvstore = 0;
        }
        enumerate(sub {
            my ($i, $param) = @_;
            my $param_arrays = $param->list_data;
            $kvstore->init($i, $param_arrays->[0]);
            $kvstore->pull($i, out => $param_arrays, priority => -$i);
        }, $self->_params);
        if($update_on_kvstore)
        {
            $kvstore->set_optimizer($self->_optimizer);
        }
        $self->_kv_store($kvstore);
        $self->_update_on_kvstore($update_on_kvstore);
    }
    else
    {
        $self->_kv_store(undef);
        $self->_update_on_kvstore(undef)
    }
    $self->_kv_initialized(1);
}

=head2 step

        Makes one step of parameter update. Should be called after
        `autograd.compute_gradient` and outside of `record()` scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
=cut

method step(Int $batch_size, Bool $ignore_stale_grad=0)
{
    if(not $self->_kv_initialized)
    {
        $self->_init_kvstore;
    }
    $self->_optimizer->rescale_grad($self->_scale/$batch_size);
    enumerate(sub {
        my ($i, $param) = @_;
        return if $param->grad_req eq 'null';
        if(not $ignore_stale_grad)
        {
            for my $data (@{ $param->list_data })
            {
                if(not $data->_fresh_grad)
                {
                    AI::MXNet::Logging->warning(
                        "Gradient of Parameter `%s` on context %s has not been updated ".
                        "by backward since last `step`. This could mean a bug in your ".
                        "model that maked it only use a subset of the Parameters (Blocks) ".
                        "for this iteration. If you are intentionally only using a subset, ".
                        "call step with ignore_stale_grad=True to suppress this ".
                        "warning and skip updating of Parameters with stale gradient",
                        $param->name, $data->context
                    );
                }
            }
        }
        if($self->_kv_store)
        {
            $self->_kv_store->push($i, $param->list_grad, priority => -$i);
            if($self->_update_on_kvstore)
            {
                $self->_kv_store->pull($i, out => $param->list_data, priority => -$i);
                return;
            }
            else
            {
                $self->_kv_store->pull($i, out => $param->list_grad, priority => -$i);
            }
        }
        for(zip($self->_updaters, $param->list_data, $param->list_grad)) {
            my ($upd, $arr, $grad) = @$_;
            if(not $ignore_stale_grad or $arr->_fresh_grad)
            {
                $upd->($i, $grad, $arr);
                $arr->_fresh_grad(0);
            }
        }
    }, $self->_params);
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
    if($self->_update_on_kvstore)
    {
        $self->_kv_store->save_optimizer_states($fname, dump_optimizer=>1);
    }
    else
    {
        open(F, ">$fname") or Carp::confess("can not open $fname: $1");
        print F $self->_updaters->[0]->get_states(dump_optimizer=>1);
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
    if($self->_update_on_kvstore)
    {
        $self->_kv_store->load_optimizer_states($fname);
        $self->_optimizer($self->_kv_store->_updater->optimizer);
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

1;
