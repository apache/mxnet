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

## TODO
## this class is here because of https://github.com/gfx/p5-Mouse/pull/67
## once 2.4.7 version of Mouse in Ubuntu for affected Perl version
## these accessors should be merged into main class

package AI::MXNet::Module::Private;
use Mouse;
has [qw/_param_names _fixed_param_names
        _aux_names _data_names _label_names _state_names
        _output_names _arg_params _aux_params
        _params_dirty _optimizer _kvstore
         _update_on_kvstore _updater _work_load_list
        _preload_opt_states _exec_group
        _data_shapes _label_shapes _context _grad_req/
] => (is => 'rw', init_arg => undef);

package AI::MXNet::Module;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use List::Util qw(max);
use Data::Dumper ();
use Mouse;

func _create_kvstore(
    Maybe[Str|AI::MXNet::KVStore] $kvstore,
    Int                           $num_device,
    HashRef[AI::MXNet::NDArray]   $arg_params
)
{
    my $update_on_kvstore = 1;
    my $kv;
    if(defined $kvstore)
    {
        if(blessed $kvstore)
        {
            $kv = $kvstore;
        }
        else
        {
            # create kvstore using the string type
            if($num_device == 1 and $kvstore !~ /dist/)
            {
                # no need to use kv for single device and single machine
            }
            else
            {
                $kv = AI::MXNet::KVStore->create($kvstore);
                if($kvstore eq 'local')
                {
                    # automatically select a proper local
                    my $max_size = max(map { product(@{ $_->shape }) } values %{ $arg_params });
                    if($max_size > 1024 * 1024 * 16)
                    {
                        $update_on_kvstore = 0;
                    }
                }
            }
        }
    }

    $update_on_kvstore = 0 if not $kv;
    return ($kv, $update_on_kvstore);
}

func _initialize_kvstore(
    AI::MXNet::KVStore           :$kvstore,
    HashRef[AI::MXNet::NDArray]  :$arg_params,
    ArrayRef[Str]                :$param_names,
    Bool                         :$update_on_kvstore,
    ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] :$param_arrays
)
{
    enumerate(sub{
        my ($idx, $param_on_devs) = @_;
        my $name = $param_names->[$idx];
        $kvstore->init($name, $arg_params->{ $name });
        if($update_on_kvstore)
        {
            $kvstore->pull($name, out => $param_on_devs, priority => -$idx);
        }
    }, $param_arrays);
}

func _update_params_on_kvstore(
    ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $param_arrays,
    ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $grad_arrays,
    AI::MXNet::KVStore           $kvstore,
    ArrayRef[Str]                $param_names
)
{
    enumerate(sub{
        my ($index, $arg_list, $grad_list) = @_;
        if(ref $grad_list eq 'ARRAY' and not defined $grad_list->[0])
        {
            return;
        }
        my $name = $param_names->[$index];
        # push gradient, priority is negative index
        $kvstore->push($name, $grad_list, priority => -$index);
        # pull back the weights
        $kvstore->pull($name, out => $arg_list, priority  => -$index);
    }, $param_arrays, $grad_arrays);
}

func _update_params(
    ArrayRef[ArrayRef[AI::MXNet::NDArray]] $param_arrays,
    ArrayRef[ArrayRef[AI::MXNet::NDArray]] $grad_arrays,
    AI::MXNet::Updater                     $updater,
    Int                                    $num_device,
    Maybe[AI::MXNet::KVStore]              $kvstore=,
    Maybe[ArrayRef[Str]]                   $param_names=
)
{
    enumerate(sub{
        my ($index, $arg_list, $grad_list) = @_;
        if(not defined $grad_list->[0])
        {
            return;
        }
        if($kvstore)
        {
            my $name = $param_names->[$index];
            # push gradient, priority is negative index
            $kvstore->push($name, $grad_list, priority => -$index);
            # pull back the sum gradients, to the same locations.
            $kvstore->pull($name, out => $grad_list, priority => -$index);
        }
        enumerate(sub {
            my ($k, $w, $g) = @_;
            # faked an index here, to make optimizer create diff
            # state for the same index but on diff devs, TODO(mli)
            # use a better solution later
            $updater->($index*$num_device+$k, $g, $w);
        }, $arg_list, $grad_list);
    }, $param_arrays, $grad_arrays);
}

method load_checkpoint(Str $prefix, Int $epoch)
{
    my $symbol = AI::MXNet::Symbol->load("$prefix-symbol.json");
    my %save_dict = %{ AI::MXNet::NDArray->load(sprintf('%s-%04d.params', $prefix, $epoch)) };
    my %arg_params;
    my %aux_params;
    while(my ($k, $v) = each %save_dict)
    {
        my ($tp, $name) = split(/:/, $k, 2);
        if($tp eq 'arg')
        {
            $arg_params{$name} = $v;
        }
        if($tp eq 'aux')
        {
            $aux_params{$name} = $v;
        }
    }
    return ($symbol, \%arg_params, \%aux_params);
}

=head1 NAME

    AI::MXNet::Module - FeedForward interface of MXNet.
    See AI::MXNet::Module::Base for the details.
=cut

extends 'AI::MXNet::Module::Base';

has '_symbol'           => (is => 'ro', init_arg => 'symbol', isa => 'AI::MXNet::Symbol', required => 1);
has '_data_names'       => (is => 'ro', init_arg => 'data_names', isa => 'ArrayRef[Str]');
has '_label_names'      => (is => 'ro', init_arg => 'label_names', isa => 'Maybe[ArrayRef[Str]]');
has 'work_load_list'    => (is => 'rw', isa => 'Maybe[ArrayRef[Int]]');
has 'fixed_param_names' => (is => 'rw', isa => 'Maybe[ArrayRef[Str]]');
has 'state_names'       => (is => 'rw', isa => 'Maybe[ArrayRef[Str]]');
has 'logger'            => (is => 'ro', default => sub { AI::MXNet::Logging->get_logger });
has '_p'                => (is => 'rw', init_arg => undef);
has 'context'           => (
    is => 'ro',
    isa => 'AI::MXNet::Context|ArrayRef[AI::MXNet::Context]',
    default => sub { AI::MXNet::Context->cpu }
);

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    if(@_%2)
    {
        my $symbol = shift;
        return $class->$orig(symbol => $symbol, @_);
    }
    return $class->$orig(@_);
};

sub BUILD
{
    my $self = shift;
    $self->_p(AI::MXNet::Module::Private->new);
    my $context = $self->context;
    if(blessed $context)
    {
        $context = [$context];
    }
    $self->_p->_context($context);
    my $work_load_list = $self->work_load_list;
    if(not defined $work_load_list)
    {
        $work_load_list = [(1)x@{$self->_p->_context}];
    }
    assert(@{ $work_load_list } == @{ $self->_p->_context });
    $self->_p->_work_load_list($work_load_list);
    my @data_names  = @{ $self->_data_names//['data'] };
    my @label_names = @{ $self->_label_names//['softmax_label'] };
    my @state_names = @{ $self->state_names//[] };
    my $arg_names   = $self->_symbol->list_arguments;
    my @input_names = (@data_names, @label_names, @state_names);
    my %input_names = map { $_ => 1 } @input_names;
    $self->_p->_param_names([grep { not exists $input_names{$_} } @{ $arg_names }]);
    $self->_p->_fixed_param_names($self->fixed_param_names//[]);
    $self->_p->_state_names(\@state_names);
    $self->_p->_aux_names($self->_symbol->list_auxiliary_states);
    $self->_p->_data_names(\@data_names);
    $self->_p->_label_names(\@label_names);
    $self->_p->_output_names($self->_symbol->list_outputs);
    $self->_p->_params_dirty(0);
    $self->_check_input_names($self->_symbol, $self->_p->_data_names, "data", 1);
    $self->_check_input_names($self->_symbol, $self->_p->_label_names, "label", 0);
    $self->_check_input_names($self->_symbol, $self->_p->_state_names, "state", 1);
    $self->_check_input_names($self->_symbol, $self->_p->_fixed_param_names, "fixed_param", 1);
}

method Module(@args) { return @args ?  __PACKAGE__->new(@args) : __PACKAGE__ }
method BucketingModule(@args) { return AI::MXNet::Module::Bucketing->new(@args) }

=head2 load

        Create a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : array ref of str
            Default is ['data'] for a typical model used in image classification.
        label_names : array ref of str
            Default is ['softmax_label'] for a typical model used in image
            classification.
        logger : Logger
            Default is AI::MXNet::Logging.
        context : Context or list of Context
            Default is cpu(0).
        work_load_list : array ref of number
            Default is undef, indicating an uniform workload.
        fixed_param_names: array ref of str
            Default is undef, indicating no network parameters are fixed.
=cut

method load(
    Str $prefix,
    Int $epoch,
    Bool $load_optimizer_states=0,
    %kwargs
)
{
    my ($sym, $args, $auxs) = __PACKAGE__->load_checkpoint($prefix, $epoch);
    my $mod = $self->new(symbol => $sym, %kwargs);
    $mod->_p->_arg_params($args);
    $mod->_p->_aux_params($auxs);
    $mod->params_initialized(1);
    if($load_optimizer_states)
    {
        $mod->_p->_preload_opt_states(sprintf('%s-%04d.states', $prefix, $epoch));
    }
    return $mod;
}

=head2 save_checkpoint

    Save current progress to a checkpoint.
    Use mx->callback->module_checkpoint as epoch_end_callback to save during training.

    Parameters
    ----------
    prefix : str
        The file prefix to checkpoint to
    epoch : int
        The current epoch number
    save_optimizer_states : bool
        Whether to save optimizer states for later training
=cut


method save_checkpoint(Str $prefix, Int $epoch, Bool $save_optimizer_states=0)
{
    $self->_symbol->save("$prefix-symbol.json");
    my $param_name = sprintf('%s-%04d.params', $prefix, $epoch);
    $self->save_params($param_name);
    AI::MXNet::Logging->info('Saved checkpoint to "%s"', $param_name);
    if($save_optimizer_states)
    {
        my $state_name = sprintf('%s-%04d.states', $prefix, $epoch);
        $self->save_optimizer_states($state_name);
        AI::MXNet::Logging->info('Saved optimizer state to "%s"', $state_name);
    }
}

=head2 model_save_checkpoint

    Checkpoint the model data into file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : AI::MXNet::Symbol
        The input symbol
    arg_params : hash ref of str to AI::MXNet::NDArray
        Model parameter, hash ref of name to AI::MXNet::NDArray of net's weights.
    aux_params : hash ref of str to NDArray
        Model parameter, hash ref of name to AI::MXNet::NDArray of net's auxiliary states.
    Notes
    -----
    - prefix-symbol.json will be saved for symbol.
    - prefix-epoch.params will be saved for parameters.
=cut

method model_save_checkpoint(
    Str                         $prefix,
    Int                         $epoch,
    Maybe[AI::MXNet::Symbol]    $symbol,
    HashRef[AI::MXNet::NDArray] $arg_params,
    HashRef[AI::MXNet::NDArray] $aux_params
)
{
    if(defined $symbol)
    {
        $symbol->save("$prefix-symbol.json");
    }
    my $param_name = sprintf('%s-%04d.params', $prefix, $epoch);
    $self->save_params($param_name, $arg_params, $aux_params);
    AI::MXNet::Logging->info('Saved checkpoint to "%s"', $param_name);
}

# Internal function to reset binded state.
method _reset_bind()
{
    $self->binded(0);
    $self->_p->_exec_group(undef);
    $self->_p->_data_shapes(undef);
    $self->_p->_label_shapes(undef);
}

method data_names()
{
    return $self->_p->_data_names;
}

method label_names()
{
    return $self->_p->_label_names;
}

method output_names()
{
    return $self->_p->_output_names;
}

method data_shapes()
{
    assert($self->binded);
    return $self->_p->_data_shapes;
}

method label_shapes()
{
    assert($self->binded);
    return $self->_p->_label_shapes;
}

method output_shapes()
{
    assert($self->binded);
    return $self->_p->_exec_group->get_output_shapes;
}

method get_params()
{
    assert($self->binded and $self->params_initialized);
    if($self->_p->_params_dirty)
    {
        $self->_sync_params_from_devices();
    }
    return ($self->_p->_arg_params, $self->_p->_aux_params);
}

method init_params(
    Maybe[AI::MXNet::Initializer]      :$initializer=AI::MXNet::Initializer->Uniform(scale => 0.01),
    Maybe[HashRef[AI::MXNet::NDArray]] :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] :$aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_init=0,
    Bool                               :$allow_extra=0
)
{
    if($self->params_initialized and not $force_init)
    {
        AI::MXNet::Logging->warning(
            "Parameters already initialized and force_init=0. "
            ."init_params call ignored."
        );
        return;
    }
    assert($self->binded, 'call bind before initializing the parameters');
    my $_impl = sub {
            my ($name, $arr, $cache) = @_;
            # Internal helper for parameter initialization
            if(defined $cache)
            {
                if(exists $cache->{$name})
                {
                    my $cache_arr = $cache->{$name};
                    # just in case the cached array is just the target itself
                    if($cache_arr->handle ne $arr->handle)
                    {
                        $cache_arr->copyto($arr);
                    }
                }
                else
                {
                    if(not $allow_missing)
                    {
                        confess("$name is not presented");
                    }
                    if(defined $initializer)
                    {
                        $initializer->($name, $arr);
                    }
                }
            }
            else
            {
                $initializer->($name, $arr) if defined $initializer;
            }
    };
    my $attrs = $self->_symbol->attr_dict;
    while(my ($name, $arr) = each %{ $self->_p->_arg_params })
    {
        $_impl->(
            AI::MXNet::InitDesc->new(
                name  => $name,
                ($attrs->{$name} ? (attrs => $attrs->{$name}) : ())
            ),
            $arr, $arg_params
        );
    }
    while(my ($name, $arr) = each %{ $self->_p->_aux_params })
    {
        $_impl->(
            AI::MXNet::InitDesc->new(
                name  => $name,
                ($attrs->{$name} ? (attrs => $attrs->{$name}) : ())
            ),
            $arr, $aux_params
        );
    }
    $self->params_initialized(1);
    $self->_p->_params_dirty(0);

    # copy the initialized parameters to devices
    $self->_p->_exec_group->set_params($self->_p->_arg_params, $self->_p->_aux_params, $allow_extra);
}

method set_params(
    HashRef[AI::MXNet::NDArray]  $arg_params,
    HashRef[AI::MXNet::NDArray]  $aux_params,
    Bool                        :$allow_missing=0,
    Bool                        :$force_init=1,
    Bool                        :$allow_extra=0
)
{
    if(not $allow_missing)
    {
        $self->init_params(
            arg_params    => $arg_params,    aux_params => $aux_params,
            allow_missing => $allow_missing, force_init => $force_init,
            allow_extra   => $allow_extra
        );
        return;
    }

    if($self->params_initialized and not $force_init)
    {
        AI::MXNet::Logging->warning(
            "Parameters already initialized and force_init=False. "
            ."set_params call ignored."
        );
        return;
    }
    $self->_p->_exec_group->set_params($arg_params, $aux_params, $allow_extra);
    $self->_p->_params_dirty(1);
    $self->params_initialized(1);
}

=head2 bind

    Bind the symbols to construct executors. This is necessary before one
    can perform computation with the module.

    Parameters
    ----------
    :$data_shapes : ArrayRef[AI::MXNet::DataDesc|NameShape]
        Typically is $data_iter->provide_data.
    :$label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
        Typically is $data_iter->provide_label.
    :$for_training : bool
        Default is 1. Whether the executors should be bind for training.
    :$inputs_need_grad : bool
        Default is 0. Whether the gradients to the input data need to be computed.
        Typically this is not needed. But this might be needed when implementing composition
        of modules.
    :$force_rebind : bool
        Default is 0. This function does nothing if the executors are already
        binded. But with this 1, the executors will be forced to rebind.
    :$shared_module : Module
        Default is undef. This is used in bucketing. When not undef, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
=cut

method bind(
    ArrayRef[AI::MXNet::DataDesc|NameShape]        :$data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]] :$label_shapes=,
    Bool                                           :$for_training=1,
    Bool                                           :$inputs_need_grad=0,
    Bool                                           :$force_rebind=0,
    Maybe[AI::MXNet::Module]                       :$shared_module=,
    GradReq|HashRef[GradReq]|ArrayRef[GradReq]     :$grad_req='write',
    Maybe[ArrayRef[Str]]                           :$state_names=$self->_p->_state_names
)
{
    # force rebinding is typically used when one want to switch from
    # training to prediction phase.
    if($force_rebind)
    {
        $self->_reset_bind();
    }
    if($self->binded)
    {
        $self->logger->warning('Already binded, ignoring bind()');
        return;
    }
    $self->for_training($for_training);
    $self->inputs_need_grad($inputs_need_grad);
    $self->binded(1);
    $self->_p->_grad_req($grad_req);

    if(not $for_training)
    {
        assert(not $inputs_need_grad);
    }
    ($data_shapes, $label_shapes) = $self->_parse_data_desc(
        $self->data_names, $self->label_names, $data_shapes, $label_shapes
    );
    $self->_p->_data_shapes($data_shapes);
    $self->_p->_label_shapes($label_shapes);
    my $shared_group;
    if($shared_module)
    {
        assert($shared_module->binded and $shared_module->params_initialized);
        $shared_group = $shared_module->_p->_exec_group;
    }

    $self->_p->_exec_group(
        AI::MXNet::DataParallelExecutorGroup->new(
            symbol            => $self->_symbol,
            contexts          => $self->_p->_context,
            workload          => $self->_p->_work_load_list,
            data_shapes       => $self->_p->_data_shapes,
            label_shapes      => $self->_p->_label_shapes,
            param_names       => $self->_p->_param_names,
            state_names       => $state_names,
            for_training      => $for_training,
            inputs_need_grad  => $inputs_need_grad,
            shared_group      => $shared_group,
            logger            => $self->logger,
            fixed_param_names => $self->_p->_fixed_param_names,
            grad_req          => $grad_req
        )
    );
    if($shared_module)
    {
        $self->params_initialized(1);
        $self->_p->_arg_params($shared_module->_p->_arg_params);
        $self->_p->_aux_params($shared_module->_p->_aux_params);
    }
    elsif($self->params_initialized)
    {
        # if the parameters are already initialized, we are re-binding
        # so automatically copy the already initialized params
        $self->_p->_exec_group->set_params($self->_p->_arg_params, $self->_p->_aux_params);
    }
    else
    {
        assert(not defined $self->_p->_arg_params and not $self->_p->_aux_params);
        my @param_arrays = (
            map { AI::MXNet::NDArray->zeros($_->[0]->shape, dtype => $_->[0]->dtype) }
            @{ $self->_p->_exec_group->_p->param_arrays }
        );
        my %arg_params;
        @arg_params{ @{ $self->_p->_param_names } } = @param_arrays;
        $self->_p->_arg_params(\%arg_params);
        my @aux_arrays = (
            map { AI::MXNet::NDArray->zeros($_->[0]->shape, dtype => $_->[0]->dtype) }
            @{ $self->_p->_exec_group->_p->aux_arrays }
        );
        my %aux_params;
        @aux_params{ @{ $self->_p->_aux_names } } = @aux_arrays;
        $self->_p->_aux_params(\%aux_params);
    }
    if($shared_module and $shared_module->optimizer_initialized)
    {
        $self->borrow_optimizer($shared_module)
    }
}

=head2 reshape

    Reshape the module for new input shapes.
    Parameters
    ----------
    :$data_shapes : ArrayRef[AI::MXNet::DataDesc]
        Typically is $data_iter->provide_data.
    :$label_shapes= : Maybe[ArrayRef[AI::MXNet::DataDesc]]
        Typically is $data_iter->provide_label.
=cut

method reshape(
    ArrayRef[AI::MXNet::DataDesc|NameShape]        :$data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]] :$label_shapes=
)
{
    assert($self->binded);
    ($data_shapes, $label_shapes) = $self->_parse_data_desc(
        $self->data_names, $self->label_names, $data_shapes, $label_shapes
    );
    $self->_p->_data_shapes($data_shapes);
    $self->_p->_label_shapes($label_shapes);
    $self->_p->_exec_group->reshape($self->_p->_data_shapes, $self->_p->_label_shapes);
}

method init_optimizer(
    Str|AI::MXNet::KVStore :$kvstore='local',
    Optimizer              :$optimizer='sgd',
    HashRef                :$optimizer_params={ learning_rate => 0.01 },
    Bool                   :$force_init=0
)
{
    assert($self->binded and $self->params_initialized);
    if($self->optimizer_initialized and not $force_init)
    {
        $self->logger->warning('optimizer already initialized, ignoring...');
        return;
    }
    if($self->_p->_params_dirty)
    {
        $self->_sync_params_from_devices;
    }

    my ($kvstore, $update_on_kvstore) = _create_kvstore(
        $kvstore,
        scalar(@{$self->_p->_context}),
        $self->_p->_arg_params
    );
    my $batch_size = $self->_p->_exec_group->_p->batch_size;
    if($kvstore and $kvstore->type =~ /dist/ and $kvstore->type =~ /_sync/)
    {
        $batch_size *= $kvstore->num_workers;
    }
    my $rescale_grad = 1/$batch_size;

    if(not blessed $optimizer)
    {
        my %idx2name;
        if($update_on_kvstore)
        {
            @idx2name{ 0..@{$self->_p->_exec_group->param_names}-1 } = @{$self->_p->_exec_group->param_names};
        }
        else
        {
            for my $k (0..@{$self->_p->_context}-1)
            {
                @idx2name{ map { $_ + $k } 0..@{$self->_p->_exec_group->param_names}-1 } = @{$self->_p->_exec_group->param_names};
            }
        }
        if(not exists $optimizer_params->{rescale_grad})
        {
            $optimizer_params->{rescale_grad} = $rescale_grad;
        }
        $optimizer = AI::MXNet::Optimizer->create(
            $optimizer,
            sym  => $self->symbol,
            param_idx2name => \%idx2name,
            %{ $optimizer_params }
        );
        if($optimizer->rescale_grad != $rescale_grad)
        {
            AI::MXNet::Logging->warning(
                "Optimizer created manually outside Module but rescale_grad "
                ."is not normalized to 1.0/batch_size/num_workers (%s vs. %s). "
                ."Is this intended?",
                $optimizer->rescale_grad, $rescale_grad
            );
        }
    }

    $self->_p->_optimizer($optimizer);
    $self->_p->_kvstore($kvstore);
    $self->_p->_update_on_kvstore($update_on_kvstore);
    $self->_p->_updater(undef);

    if($kvstore)
    {
        # copy initialized local parameters to kvstore
        _initialize_kvstore(
            kvstore           => $kvstore,
            param_arrays      => $self->_p->_exec_group->_p->param_arrays,
            arg_params        => $self->_p->_arg_params,
            param_names       => $self->_p->_param_names,
            update_on_kvstore => $update_on_kvstore
        );
    }
    if($update_on_kvstore)
    {
        $kvstore->set_optimizer($self->_p->_optimizer);
    }
    else
    {
        $self->_p->_updater(AI::MXNet::Optimizer->get_updater($optimizer));
    }
    $self->optimizer_initialized(1);

    if($self->_p->_preload_opt_states)
    {
        $self->load_optimizer_states($self->_p->_preload_opt_states);
        $self->_p->_preload_opt_states(undef);
    }
}

=head2 borrow_optimizer

    Borrow optimizer from a shared module. Used in bucketing, where exactly the same
    optimizer (esp. kvstore) is used.

    Parameters
    ----------
    shared_module : AI::MXNet::Module
=cut

method borrow_optimizer(AI::MXNet::Module $shared_module)
{
    assert($shared_module->optimizer_initialized);
    $self->_p->_optimizer($shared_module->_p->_optimizer);
    $self->_p->_kvstore($shared_module->_p->_kvstore);
    $self->_p->_update_on_kvstore($shared_module->_p->_update_on_kvstore);
    $self->_p->_updater($shared_module->_p->_updater);
    $self->optimizer_initialized(1);
}

method forward(
    AI::MXNet::DataBatch $data_batch,
    Maybe[Bool]         :$is_train=
)
{
    assert($self->binded and $self->params_initialized);

    my @curr_data_shapes = map { $_->shape } @{ $self->data_shapes };
    my @new_data_shapes  = map { $_->shape } @{ $data_batch->data };
    if(Data::Dumper->Dump(\@curr_data_shapes) ne Data::Dumper->Dump(\@new_data_shapes))
    {
        my $new_dshape;
        if($data_batch->can('provide_data') and $data_batch->provide_data)
        {
            $new_dshape = $data_batch->provide_data;
        }
        else
        {
            $new_dshape = [];
            for(zip($self->data_shapes, \@new_data_shapes)) {
                my ($i, $shape) = @$_;
                push @{ $new_dshape }, AI::MXNet::DataDesc->new(
                    $i->name, $shape, $i->dtype, $i->layout
                );
            }
        }
        my $new_lshape;
        if($data_batch->can('provide_label') and $data_batch->provide_label)
        {
            $new_lshape = $data_batch->provide_label;
        }
        elsif($data_batch->can('label') and $data_batch->label)
        {
            $new_lshape = [];
            for(zip($self->label_shapes, $data_batch->label)) {
                my ($i, $j) = @$_;
                push @{ $new_lshape }, AI::MXNet::DataDesc->new(
                    $i->name, $j->shape, $i->dtype, $i->layout
                );
            }
        }
        $self->reshape(data_shapes => $new_dshape, label_shapes => $new_lshape);
    }
    $self->_p->_exec_group->forward($data_batch, $is_train);
}

method backward(Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]] $out_grads=)
{
    assert($self->binded and $self->params_initialized);
    $self->_p->_exec_group->backward($out_grads);
}

method update()
{
    assert($self->binded and $self->params_initialized and $self->optimizer_initialized);
    $self->_p->_params_dirty(1);
    if($self->_p->_update_on_kvstore)
    {
        _update_params_on_kvstore(
            $self->_p->_exec_group->_p->param_arrays,
            $self->_p->_exec_group->_p->grad_arrays,
            $self->_p->_kvstore,
            $self->_p->_exec_group->param_names
        );
    }
    else
    {
        _update_params(
            $self->_p->_exec_group->_p->param_arrays,
            $self->_p->_exec_group->_p->grad_arrays,
            $self->_p->_updater,
            scalar(@{ $self->_p->_context}),
            $self->_p->_kvstore,
            $self->_p->_exec_group->param_names
        );
    }
}

method get_outputs(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    return $self->_p->_exec_group->get_outputs($merge_multi_context);
}

method get_input_grads(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized and $self->inputs_need_grad);
    return $self->_p->_exec_group->get_input_grads($merge_multi_context);
}

method get_states(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    return $self->_p->_exec_group->get_states($merge_multi_context);
}

method set_states(:$states=, :$value=)
{
    assert($self->binded and $self->params_initialized);
    return $self->_p->_exec_group->set_states($states, $value);
}

method update_metric(
    AI::MXNet::EvalMetric $eval_metric,
    ArrayRef[AI::MXNet::NDArray] $labels
)
{
    $self->_p->_exec_group->update_metric($eval_metric, $labels);
}

=head2 _sync_params_from_devices

    Synchronize parameters from devices to CPU. This function should be called after
    calling 'update' that updates the parameters on the devices, before one can read the
    latest parameters from $self->_arg_params and $self->_aux_params.
=cut

method _sync_params_from_devices()
{
    $self->_p->_exec_group->get_params($self->_p->_arg_params, $self->_p->_aux_params);
    $self->_p->_params_dirty(0);
}

method save_optimizer_states(Str $fname)
{
    assert($self->optimizer_initialized);
    if($self->_p->_update_on_kvstore)
    {
        $self->_p->_kvstore->save_optimizer_states($fname);
    }
    else
    {
        open(F, ">:raw", "$fname") or confess("can't open $fname for writing: $!");
        print F $self->_p->_updater->get_states();
        close(F);
    }
}

method load_optimizer_states(Str $fname)
{
    assert($self->optimizer_initialized);
    if($self->_p->_update_on_kvstore)
    {
        $self->_p->_kvstore->load_optimizer_states($fname);
    }
    else
    {
        open(F, "<:raw", "$fname") or confess("can't open $fname for reading: $!");
        my $data;
        { local($/) = undef; $data = <F>; }
        close(F);
        $self->_p->_updater->set_states($data);
    }
}

method install_monitor(AI::MXNet::Monitor $mon)
{
    assert($self->binded);
    $self->_p->_exec_group->install_monitor($mon);
}

method _updater()
{
    $self->_p->_updater;
}

method _kvstore()
{
    $self->_p->_kvstore;
}

1;
