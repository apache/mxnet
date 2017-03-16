package AI::MXNet::Module::Bucketing;
use Mouse;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;

=encoding UTF-8

=head1 NAME

AI::MXNet::Module::Bucketing

=head1 SYNOPSIS

    This is alpha release.
    Please refer to t dir for examples.

=head1 DESCRIPTION

    Implements the AI::MXNet::Module::Base API, and allows multiple
symbols to be used depending on the `bucket_key` provided by each different
mini-batch of data
=cut


=head2 new

    Parameters
    ----------
    sym_gen : subref or any perl object that overloads &{} op
        A sub when called with a bucket key, returns a list with triple
        of ($symbol, $data_names, $label_names).
    default_bucket_key : str or anything else
        The key for the default bucket.
    logger : Logger
    context : Context or arrayref of Context
        Default `cpu()`
    work_load_list : array ref of Num
        Default undef, indicating uniform workload.
    fixed_param_names: arrayref of str
        Default undef, indicating no network parameters are fixed.
    state_names : arrayref of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to 0 and can be set by set_states()

=cut

extends 'AI::MXNet::Module::Base';
has '_sym_gen'            => (is => 'ro', init_arg => 'sym_gen', required => 1);
has '_default_bucket_key' => (is => 'rw', init_arg => 'default_bucket_key', required => 1);
has '_context'            => (
    is => 'ro', isa => 'AI::MXNet::Context|ArrayRef[AI::MXNet::Context]',
    lazy => 1, default => sub { AI::MXNet::Context->cpu },
    init_arg => 'context'
);
has '_work_load_list'     => (is => 'rw', init_arg => 'work_load_list', isa => 'ArrayRef[Num]');
has '_curr_module'        => (is => 'rw', init_arg => undef);
has '_buckets'            => (is => 'rw', init_arg => undef, default => sub { +{} });
has '_fixed_param_names'  => (is => 'rw', isa => 'ArrayRef[Str]', init_arg => 'fixed_param_names');
has '_state_names'        => (is => 'rw', isa => 'ArrayRef[Str]', init_arg => 'state_names');
has '_params_dirty'       => (is => 'rw', init_arg => undef);

sub BUILD
{
    my ($self, $original_params) = @_;
    $self->_fixed_param_names([]) unless defined $original_params->{fixed_param_names};
    $self->_state_names([]) unless defined $original_params->{state_names};
    $self->_params_dirty(0);
    my ($symbol, $data_names, $label_names) = &{$self->_sym_gen}($self->_default_bucket_key);
    $self->_check_input_names($symbol, $data_names//[], "data", 1);
    $self->_check_input_names($symbol, $label_names//[], "label", 0);
    $self->_check_input_names($symbol, $self->_state_names, "state", 1);
    $self->_check_input_names($symbol, $self->_fixed_param_names, "fixed_param", 1);
}

method _reset_bind()
{
    $self->binded(0);
    $self->_buckets({});
    $self->_curr_module(undef);
}

=head2 data_names

    A list of names for data required by this module.
=cut

method data_names()
{
    if($self->binded)
    {
        return $self->_curr_module->data_names;
    }
    else
    {
        return (&{$self->_sym_gen}($self->_default_bucket_key))[1];
    }
}

=head2 output_names

    A list of names for the outputs of this module.
=cut

method output_names()
{
    if($self->binded)
    {
        return $self->_curr_module->ouput_names;
    }
    else
    {
        my ($symbol) = &{$self->_sym_gen}($self->_default_bucket_key);
        return $symbol->list_ouputs;
    }
}

=head2 data_shapes

        Get data shapes.
        Returns
        -------
        An array ref of AI::MXNet::DataDesc objects.
=cut

method data_shapes()
{
    assert($self->binded);
    return $self->_curr_module->data_shapes;
}

=head2 label_shapes

        Get label shapes.
        Returns
        -------
        An array ref of AI::MXNet::DataDesc objects. The return value could be undef if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        An array ref of AI::MXNet::DataDesc objects.
=cut

method label_shapes()
{
    assert($self->binded);
    return $self->_curr_module->label_shapes;
}

=head2 output_shapes

        Get output shapes.
        Returns
        -------
        An array ref of AI::MXNet::DataDesc objects.
=cut

method output_shapes()
{
    assert($self->binded);
    return $self->_curr_module->output_shapes;
}

=head2 get_params

        Get current parameters.
        Returns
        -------
        List of ($arg_params, $aux_params), each a hash ref of name to parameters (
        AI::MXNet::NDArray) mapping.
=cut

method get_params()
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->_params_dirty($self->_params_dirty);
    my $params = $self->_curr_module->get_params;
    $self->_params_dirty(0);
    return $params;
}

=head2 set_params

        Assign parameter and aux state values.

        Parameters
        ----------
        arg_params : HashRef[AI::MXNet::NDArray]
        aux_params : HashRef[AI::MXNet::NDArray]
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
=cut

method set_params(
    HashRef[AI::MXNet::NDArray] $arg_params,
    HashRef[AI::MXNet::NDArray] $aux_params,
    Bool                        $allow_missing=0,
    Bool                        $force_init=1
)
{
    if(not $allow_missing)
    {
        $self->init_params(
            arg_params    => $arg_params,    aux_params => $aux_params,
            allow_missing => $allow_missing, force_init => $force_init
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
    $self->_curr_module->set_params(
        $arg_params, $aux_params,
        allow_missing => $allow_missing,
        force_init    => $force_init
    );
    # because we didn't update self._arg_params, they are dirty now.
    $self->_params_dirty(1);
    $self->params_initialized(1);
}

=head2 init_params

        Initialize parameters.

        Parameters
        ----------
        initializer : AI::MXNet::Initializer, default AI::MXNet::Initalizer->Uniform->(scale => 0.01)
        arg_params : HashRef
            Default undef. Existing parameters. This has higher priority than `initializer`.
        aux_params : HashRef
            Default undef. Existing auxiliary states. This has higher priority than `initializer`.
        allow_missing : Bool
            Allow missing values in `arg_params` and `aux_params` (if not undef). In this case,
            missing values will be filled with `initializer` Default 0
        force_init : Bool
            Default 0
=cut

method init_params(
    AI::MXNet::Initializer             :$initializer=AI::MXNet::Initializer->Uniform(scale => 0.01),
    Maybe[HashRef[AI::MXNet::NDArray]] :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] :$aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_init=0
)
{
    return if(not $self->params_initialized and not $force_init);
    assert($self->binded, 'call bind before initializing the parameters');
    $self->_curr_module->init_params(
        initializer   => $initializer,
        arg_params    => $arg_params,
        aux_params    => $aux_params,
        allow_missing => $allow_missing,
        force_init    => $force_init
    );
    $self->_params_dirty(0);
    $self->params_initialized(1);
}

method get_states(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->get_states($merge_multi_context);
}

method set_states($states, $value)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->set_states($states, $value);
}

=head2 bind

        Binding for a AI::MXNet::Module::Bucketing means setting up the buckets and bind the
        executor for the default bucket key. Executors corresponding to other keys are
        binded afterwards with 'switch_bucket'.

        Parameters
        ----------
        data_shapes : ArrayRef[AI::MXNet::DataDesc|NameShape]
            This should correspond to the symbol for the default bucket.
        label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
            This should correspond to the symbol for the default bucket.
        for_training : Bool
            Default is 1.
        inputs_need_grad : Bool
            Default is 0.
        force_rebind : Bool
            Default is 0.
        shared_module : AI::MXNet::Module::Bucketing
            Default is undef. This value is currently not used.
        grad_req : str, array ref of str, hash ref of str to str
            Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
            (default to 'write').
            Can be specified globally (str) or for each argument (array ref, hash ref).
        bucket_key : str
            bucket key for binding. by default use the default_bucket_key
=cut

method bind(
    ArrayRef[AI::MXNet::DataDesc|NameShape]                    $data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]            :$label_shapes=,
    Bool                                                      :$for_training=1,
    Bool                                                      :$inputs_need_grad=0,
    Bool                                                      :$force_rebind=0,
    Maybe[AI::MXNet::BaseModule]                              :$shared_module=,
    Str|ArrayRef[Str]|HashRef[Str]                            :$grad_req='write',
    Str                                                       :$bucket_key
)
{
    # in case we already initialized params, keep it
    my ($arg_params, $aux_params);
    if($self->params_initialized)
    {
        ($arg_params, $aux_params) = $self->get_params;
    }

    # force rebinding is typically used when one want to switch from
    # training to prediction phase.
    $self->_reset_bind if $force_rebind;

    if($self->binded)
    {
        $self->logger->warning('Already binded, ignoring bind()');
        return;
    }

    assert((not defined $shared_module), 'shared_module for BucketingModule is not supported');

    $self->for_training($for_training);
    $self->inputs_need_grad($inputs_need_grad);
    $self->binded(1);

    my ($symbol, $data_names, $label_names) = &{$self->_sym_gen}($bucket_key//$self->_default_bucket_key);
    my $module = AI::MXNet::Module->new(
            symbol            => $symbol,
            data_names        => $data_names,
            label_names       => $label_names,
            logger            => $self->logger,
            context           => $self->_context,
            work_load_list    => $self->_work_load_list,
            state_names       => $self->_state_names,
            fixed_param_names => $self->_fixed_param_names
    );
    $module->bind(
        data_shapes      => $data_shapes,
        label_shapes     => $label_shapes,
        for_training     => $for_training,
        inputs_need_grad => $inputs_need_grad,
        force_rebind     => 0,
        shared_module    => undef,
        grad_req         => $grad_req
    );
    $self->_curr_module($module);
    $self->_buckets->{ $self->_default_bucket_key } = $module;

    # copy back saved params, if already initialized
    if($self->params_initialized)
    {
        $self->set_params($arg_params, $aux_params);
    }
}

=head2 switch_bucket

        Switch to a different bucket. This will change $self->_curr_module.

        Parameters
        ----------
        bucket_key : str (or any perl object that overloads "" op)
            The key of the target bucket.
        data_shapes : ArrayRef[AI::MXNet::DataDesc|NameShape]
            Typically `data_batch.provide_data`.
        label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
            Typically `data_batch.provide_label`.
=cut

method switch_bucket(
    ArrayRef[AI::MXNet::DataDesc|NameShape]                   :$data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]            :$label_shapes=,
                                                              :$bucket_key
)
{
    assert($self->binded, 'call bind before switching bucket');
    if(not exists $self->_buckets->{ $bucket_key })
    {
        my ($symbol, $data_names, $label_names) = &{$self->_sym_gen}($bucket_key);
        my $module = AI::MXNet::Module->new(
            symbol         => $symbol,
            data_names     => $data_names,
            label_names    => $label_names,
            logger         => $self->logger,
            context        => $self->_context,
            work_load_list => $self->_work_load_list
        );
        $module->bind(
            data_shapes      => $data_shapes,
            label_shapes     => $label_shapes,
            for_training     => $self->_curr_module->for_training,
            inputs_need_grad => $self->_curr_module->inputs_need_grad,
            force_rebind     => 0,
            shared_module    => $self->_buckets->{ $self->_default_bucket_key },
        );
        $self->_curr_module($module);
        $self->_buckets->{ $bucket_key } = $module;
    }
}

=head2  init_optimizer

        Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or AI::MXNet::KVStore object
            Default 'local'
        optimizer : str or AI::MXNet::Optimizer object
            Default 'sgd'
        optimizer_params : hash ref
            Default: { learning_rate =>  0.01 }
        force_init : Bool
            Default 0, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
=cut

method init_optimizer(
    Str        :$kvstore='local',
    Optimizer  :$optimizer='sgd',
    HashRef    :$optimizer_params={ learning_rate => 0.01 },
    Bool       :$force_init=0
)
{
    assert($self->binded and $self->params_initialized);
    if($self->optimizer_initialized and not $force_init)
    {
        $self->logger->warning('optimizer already initialized, ignoring.');
        return;
    }

    $self->_curr_module->init_optimizer(
        kvstore           => $kvstore,
        optimizer         => $optimizer,
        optimizer_params  => $optimizer_params,
        force_init        => $force_init
    );
    for my $mod (values %{ $self->_buckets })
    {
        if($mod ne $self->_curr_module)
        {
            $mod->borrow_optimizer($self->_curr_module);
        }
    }
    $self->optimizer_initialized(1);
}

=head2 forward

        Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
        is_train : Bool
            Default is undef, in which case 'is_train' is taken from $self->for_training.
=cut

method forward(
    AI::MXNet::DataBatch  $data_batch,
    Bool                 :$is_train=
)
{
    assert($self->binded and $self->params_initialized);
    $self->switch_bucket(
        bucket_key   => $data_batch->bucket_key,
        data_shapes  => $data_batch->provide_data,
        label_shapes => $data_batch->provide_label
    );
    $self->_curr_module->forward($data_batch, is_train => $is_train);
}

=head2 backward

        Backward computation.
        Parameters
        ----------
        out_grads : Maybe[ArrayRef[AI::MXNet::NDArray]|AI::MXNet::NDArray]
        Default: undef
=cut

method backward(Maybe[ArrayRef[AI::MXNet::NDArray]|AI::MXNet::NDArray] $out_grads=)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->backward($out_grads);
}

=head2 update

        Update parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
=cut

method update()
{
    assert($self->binded and $self->params_initialized and $self->optimizer_initialized);
    $self->_params_dirty(1);
    $self->_curr_module->update;
}

=head2 get_outputs

        Get outputs from a previous forward computation.

        Parameters
        ----------
        merge_multi_context : Bool
            Default is 1. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A 1 value indicates that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If 'merge_multi_context' is 1, it is like [out1, out2]. Otherwise, it
        is like [[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]. All the output
        elements are pdl objects.
=cut


method get_outputs(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    return $self->_curr_module->get_outputs($merge_multi_context);
}

=head2 get_input_grads

        Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : Bool
            Default is 1. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A 1 value indicates that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If 'merge_multi_context' is 1, it is like [grad1, grad2]. Otherwise, it
        is like [[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]. All the output
        elements are AI::MXNet::NDArray objects.
=cut

method get_input_grads(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized and $self->inputs_need_grad);
    return $self->_curr_module->get_input_grads($merge_multi_context);
}

=head2 update_metric
        Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : AI::MXNet::EvalMetric
        labels : ArrayRef[AI::MXNet::NDArray]
            Typically $data_batch->label.
=cut

method update_metric(
    AI::MXNet::EvalMetric $eval_metric,
    ArrayRef[AI::MXNet::NDArray] $labels
)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->update_metric($eval_metric, $labels);
}

=head2 symbol

    The symbol of the current bucket being used.
=cut

method symbol()
{
    assert($self->binded);
    return $self->_curr_module->symbol;
}

=head2 install_monitor

        Install monitor on all executors.

        Paramters
        ---------
        AI::MXNet::Monitor
=cut

method install_monitor(AI::MXNet::Monitor $mon)
{
    assert($self->binded);
    for my $mod (values %{ $self->_buckets })
    {
        $mod->install_monitor($mon);
    }
}

1;
