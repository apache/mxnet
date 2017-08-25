package AI::MXNet::BatchEndParam;
use Mouse;
use AI::MXNet::Function::Parameters;
has [qw/epoch nbatch/] => (is => 'rw', isa => 'Int');
has 'eval_metric'      => (is => 'rw', isa => 'AI::MXNet::EvalMetric');

package AI::MXNet::Module::Base;
use Mouse;
use AI::MXNet::Base;
use Time::HiRes qw(time);

=head1 NAME

    AI::MXNet::Module::Base - Base class for AI::MXNet::Module and AI::MXNet::Module::Bucketing
=cut

func _as_list($obj)
{
    return [$obj] if ((ref($obj)//'') ne 'ARRAY');
    return $obj;
}

# Check that all input names are in symbol's argument
method _check_input_names(
    AI::MXNet::Symbol $symbol,
    ArrayRef[Str]     $names,
    Str               $typename,
    Bool              $throw
)
{
    my @candidates;
    my %args = map {
        push @candidates, $_ if not /_(?:weight|bias|gamma|beta)$/;
        $_ => 1
    } @{ $symbol->list_arguments };
    for my $name (@$names)
    {
        my $msg;
        if(not exists $args{$name} and $name ne 'softmax_label')
        {
            $msg = sprintf("\033[91mYou created Module with Module(..., %s_names=%s) but "
                ."input with name '%s' is not found in symbol.list_arguments(). "
                ."Did you mean one of:\n\t%s\033[0m",
                $typename, "@$names", $name, join("\n\t", @candidates)
            );
            if($throw)
            {
                confess($msg);
            }
            else
            {
                AI::MXNet::Logging->warning($msg);
            }
        }
    }
}

# Check that input names matches input data descriptors
method _check_names_match(
    ArrayRef[Str]                  $data_names,
    ArrayRef[NameShapeOrDataDesc]  $data_shapes,
    Str                            $name,
    Bool                           $throw
)
{
    return if (not @$data_shapes and @$data_names == 1 and  $data_names->[0] eq 'softmax_label');
    my @actual = map { @{$_}[0] } @{ $data_shapes };
    if("@$data_names" ne "@actual")
    {
        my $msg = sprintf(
            "Data provided by %s_shapes don't match names specified by %s_names (%s vs. %s)",
            $name, $name, "@$data_shapes", "@$data_names"
        );
        if($throw)
        {
            confess($msg);
        }
        else
        {
            AI::MXNet::Logging->warning($msg);
        }
    }
}

method _parse_data_desc(
    ArrayRef[Str]                                  $data_names,
    Maybe[ArrayRef[Str]]                           $label_names,
    ArrayRef[NameShapeOrDataDesc]                  $data_shapes,
    Maybe[ArrayRef[NameShapeOrDataDesc]]           $label_shapes
)
{
    $data_shapes = [map { blessed $_ ? $_ : AI::MXNet::DataDesc->new(@$_) } @$data_shapes];
    $self->_check_names_match($data_names, $data_shapes, 'data', 1);
    if($label_shapes)
    {
        $label_shapes = [map { blessed $_ ? $_ : AI::MXNet::DataDesc->new(@$_) } @$label_shapes];
        $self->_check_names_match($label_names, $label_shapes, 'label', 0);
    }
    else
    {
        $self->_check_names_match($label_names, [], 'label', 0);
    }
    return ($data_shapes, $label_shapes);
}

=head1 DESCRIPTION

    The base class of a modules. A module represents a computation component. The design
    purpose of a module is that it abstract a computation "machine", that one can run forward,
    backward, update parameters, etc. We aim to make the APIs easy to use, especially in the
    case when we need to use imperative API to work with multiple modules (e.g. stochastic
    depth network).

    A module has several states:

        - Initial state. Memory is not allocated yet, not ready for computation yet.
        - Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated,
        ready for computation.
        - Parameter initialized. For modules with parameters, doing computation before initializing
        the parameters might result in undefined outputs.
        - Optimizer installed. An optimizer can be installed to a module. After this, the parameters
        of the module can be updated according to the optimizer after gradients are computed
        (forward-backward).

    In order for a module to interact with others, a module should be able to report the
    following information in its raw stage (before binded)

        - data_names: array ref of string indicating the names of required data.
        - output_names: array ref of string indicating the names of required outputs.

    And also the following richer information after binded:

    - state information
        - binded: bool, indicating whether the memory buffers needed for computation
        has been allocated.
        - for_training: whether the module is binded for training (if binded).
        - params_initialized: bool, indicating whether the parameters of this modules
        has been initialized.
        - optimizer_initialized: bool, indicating whether an optimizer is defined
        and initialized.
        - inputs_need_grad: bool, indicating whether gradients with respect to the
        input data is needed. Might be useful when implementing composition of modules.

    - input/output information
        - data_shapes: am array ref of [name, shape]. In theory, since the memory is allocated,
        we could directly provide the data arrays. But in the case of data parallelization,
        the data arrays might not be of the same shape as viewed from the external world.
        - label_shapes: an array ref of [name, shape]. This might be [] if the module does
        not need labels (e.g. it does not contains a loss function at the top), or a module
        is not binded for training.
        - output_shapes: an array ref of [name, shape] for outputs of the module.

    - parameters (for modules with parameters)
        - get_params(): return an array ($arg_params, $aux_params). Each of those
        is a hash ref of name to NDArray mapping. Those NDArrays always on
        CPU. The actual parameters used for computing might be on other devices (GPUs),
        this function will retrieve (a copy of) the latest parameters. Therefore, modifying
        - get_params($arg_params, $aux_params): assign parameters to the devices
        doing the computation.
        - init_params(...): a more flexible interface to assign or initialize the parameters.

    - setup
        - bind(): prepare environment for computation.
        - init_optimizer(): install optimizer for parameter updating.

    - computation
        - forward(data_batch): forward operation.
        - backward(out_grads=): backward operation.
        - update(): update parameters according to installed optimizer.
        - get_outputs(): get outputs of the previous forward operation.
        - get_input_grads(): get the gradients with respect to the inputs computed
        in the previous backward operation.
        - update_metric(metric, labels): update performance metric for the previous forward
        computed results.

    - other properties (mostly for backward compatability)
        - symbol: the underlying symbolic graph for this module (if any)
        This property is not necessarily constant. For example, for AI::MXNet::Module::Bucketing,
        this property is simply the *current* symbol being used. For other modules,
        this value might not be well defined.

    When those intermediate-level API are implemented properly, the following
    high-level API will be automatically available for a module:

        - fit: train the module parameters on a data set
        - predict: run prediction on a data set and collect outputs
        - score: run prediction on a data set and evaluate performance
=cut

has 'logger'            => (is => 'rw', default => sub { AI::MXNet::Logging->get_logger });
has '_symbol'           => (is => 'rw', init_arg => 'symbol', isa => 'AI::MXNet::Symbol');
has [
    qw/binded for_training inputs_need_grad
    params_initialized optimizer_initialized/
]                       => (is => 'rw', isa => 'Bool', init_arg => undef, default => 0);

################################################################################
# High Level API
################################################################################

=head2 forward_backward

    A convenient function that calls both forward and backward.
=cut

method forward_backward(AI::MXNet::DataBatch $data_batch)
{
    $self->forward($data_batch, is_train => 1);
    $self->backward();
}

=head2 score

    Run prediction on eval_data and evaluate the performance according to
    eval_metric.

    Parameters
    ----------
    $eval_data   : AI::MXNet::DataIter
    $eval_metric : AI::MXNet::EvalMetric
    :$num_batch= : Maybe[Int]
        Number of batches to run. Default is undef, indicating run until the AI::MXNet::DataIter
        finishes.
    :$batch_end_callback= : Maybe[Callback]
        Could also be a array ref of functions.
    :$reset=1 : Bool
        Default 1, indicating whether we should reset $eval_data before starting
        evaluating.
    $epoch=0 : Int
        Default is 0. For compatibility, this will be passed to callbacks (if any). During
        training, this will correspond to the training epoch number.
=cut

method score(
    AI::MXNet::DataIter                 $eval_data,
    EvalMetric                          $eval_metric,
    Maybe[Int]                          :$num_batch=,
    Maybe[Callback]|ArrayRef[Callback]  :$batch_end_callback=,
    Maybe[Callback]|ArrayRef[Callback]  :$score_end_callback=,
    Bool                                :$reset=1,
    Int                                 :$epoch=0
)
{
    assert($self->binded and $self->params_initialized);
    $eval_data->reset if $reset;
    if(not blessed $eval_metric or not $eval_metric->isa('AI::MXNet::EvalMetric'))
    {
        $eval_metric = AI::MXNet::Metric->create($eval_metric);
    }

    $eval_metric->reset();
    my $actual_num_batch = 0;
    my $nbatch = 0;
    while(my $eval_batch = <$eval_data>)
    {
        last if (defined $num_batch and $nbatch == $num_batch);
        $self->forward($eval_batch, is_train => 0);
        $self->update_metric($eval_metric, $eval_batch->label);

        if (defined $batch_end_callback)
        {
            my $batch_end_params = AI::MXNet::BatchEndParam->new(
                epoch  => $epoch,
                nbatch => $nbatch,
                eval_metric => $eval_metric
            );
            for my $callback (@{ _as_list($batch_end_callback) })
            {
                &{$callback}($batch_end_params);
            }
        }
        $actual_num_batch++;
        $nbatch++
    }
    if($score_end_callback)
    {
        my $params = AI::MXNet::BatchEndParam->new(
            epoch  => $epoch,
            nbatch => $actual_num_batch,
            eval_metric => $eval_metric,
        );
        for my $callback (@{ _as_list($score_end_callback) })
        {
            &{callback}($params);
        }
    }
    return $eval_metric->get_name_value;
}

=head2  iter_predict

    Iterate over predictions.

    Parameters
    ----------
    $eval_data : AI::MXNet::DataIter
    :$num_batch= : Maybe[Int]
        Default is undef, indicating running all the batches in the data iterator.
    :$reset=1 : bool
        Default is 1, indicating whether we should reset the data iter before start
        doing prediction.
=cut

method iter_predict(AI::MXNet::DataIter $eval_data, Maybe[Int] :$num_batch=, Bool :$reset=1)
{
    assert($self->binded and $self->params_initialized);
    if($reset)
    {
        $eval_data->reset;
    }
    my $nbatch = 0;
    my @out;
    while(my $eval_batch = <$eval_data>)
    {
        last if defined $num_batch and $nbatch == $num_batch;
        $self->forward($eval_batch, is_train => 0);
        my $pad = $eval_batch->pad;
        my $outputs = [
            map { $_->slice([0, $_->shape->[0] - ($pad//0) - 1]) } @{ $self->get_outputs() }
        ];
        push @out, [$outputs, $nbatch, $eval_batch];
        $nbatch++;
    }
    return @out;
}

=head2 predict

    Run prediction and collect the outputs.

    Parameters
    ----------
    $eval_data  : AI::MXNet::DataIter
    :$num_batch= : Maybe[Int]
        Default is undef, indicating running all the batches in the data iterator.
    :$merge_batches=1 : Bool
        Default is 1.
    :$reset=1 : Bool
        Default is 1, indicating whether we should reset the data iter before start
        doing prediction.
    :$always_output_list=0 : Bool
    Default is 0, see the doc for return values.

    Returns
    -------
    When $merge_batches is 1 (by default), the return value will be an array ref
    [$out1, $out2, $out3] where each element is concatenation of the outputs for
    all the mini-batches. If $always_output_list` also is 0 (by default),
    then in the case of a single output, $out1 is returned in stead of [$out1].

    When $merge_batches is 0, the return value will be a nested array ref like
    [[$out1_batch1, $out2_batch1], [$out1_batch2], ...]. This mode is useful because
    in some cases (e.g. bucketing), the module does not necessarily produce the same
    number of outputs.

    The objects in the results are AI::MXNet::NDArray`s. If you need to work with pdl array,
    just call ->aspdl() on each AI::MXNet::NDArray.
=cut

method predict(
    AI::MXNet::DataIter $eval_data,
    Maybe[Int] :$num_batch=, Bool :$merge_batches=1, Bool :$reset=1, Bool :$always_output_list=0
)
{
    assert($self->binded and $self->params_initialized);
    $eval_data->reset() if $reset;

    my @output_list;
    my $nbatch = 0;
    while(my $eval_batch = <$eval_data>)
    {
        last if defined $num_batch and $nbatch == $num_batch;
        $self->forward($eval_batch, is_train => 0);
        my $pad = $eval_batch->pad;
        my $outputs = [map { $_->slice([0, $_->shape->[0]-($pad//0)-1])->copy } @{ $self->get_outputs }];
        push @output_list, $outputs;
    }
    return () unless @output_list;
    if($merge_batches)
    {
        my $num_outputs = @{ $output_list[0] };
        for my $out (@output_list)
        {
            unless(@{ $out } == $num_outputs)
            {
                confess('Cannot merge batches, as num of outputs is not the same '
                       .'in mini-batches. Maybe bucketing is used?');
            }
        }
        my @output_list2;
        for my $i (0..$num_outputs-1)
        {
            push @output_list2,
                 AI::MXNet::NDArray->concatenate([map { $_->[$i] } @output_list]);
        }
        if($num_outputs == 1 and not $always_output_list)
        {
            return $output_list2[0];
        }
        return @output_list2;
    }
    return @output_list;
}

=head2 fit

    Train the module parameters.

    Parameters
    ----------
    $train_data : AI::MXNet::DataIter
    :$eval_data= : Maybe[AI::MXNet::DataIter]
        If not undef, it will be used as a validation set to evaluate the performance
        after each epoch.
    :$eval_metric='acc' : str or AI::MXNet::EvalMetric subclass object.
        Default is 'accuracy'. The performance measure used to display during training.
        Other possible predefined metrics are:
        'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'
    :$epoch_end_callback= : Maybe[Callback]|ArrayRef[Callback] function or array ref of functions.
        Each callback will be called with the current $epoch, $symbol, $arg_params
        and $aux_params.
    :$batch_end_callback= : Maybe[Callback]|ArrayRef[Callback] function or array ref of functions.
        Each callback will be called with a AI::MXNet::BatchEndParam.
    :$kvstore='local' : str or AI::MXNet::KVStore
        Default is 'local'.
    :$optimizer : str or AI::MXNet::Optimizer
        Default is 'sgd'
    :$optimizer_params : hash ref
        Default { learning_rate => 0.01 }.
        The parameters for the optimizer constructor.
    :$eval_end_callback= : Maybe[Callback]|ArrayRef[Callback] function or array ref of functions
        These will be called at the end of each full evaluation, with the metrics over
        the entire evaluation set.
    :$eval_batch_end_callback : Maybe[Callback]|ArrayRef[Callback] function or array ref of functions
        These will be called at the end of each minibatch during evaluation
    :$initializer= : Initializer
        Will be called to initialize the module parameters if not already initialized.
    :$arg_params= : hash ref
        Default undef, if not undef, must be an existing parameters from a trained
        model or loaded from a checkpoint (previously saved model). In this case,
        the value here will be used to initialize the module parameters, unless they
        are already initialized by the user via a call to init_params or fit.
        $arg_params have higher priority than the $initializer.
    :$aux_params= : hash ref
        Default is undef. This is similar to the $arg_params, except for auxiliary states.
    :$allow_missing=0 : Bool
        Default is 0. Indicates whether we allow missing parameters when $arg_params
        and $aux_params are not undefined. If this is 1, then the missing parameters
        will be initialized via the $initializer.
    :$force_rebind=0 : Bool
        Default is 0. Whether to force rebinding the executors if already binded.
    :$force_init=0 : Bool
        Default is 0. Indicates whether we should force initialization even if the
        parameters are already initialized.
    :$begin_epoch=0 : Int
        Default is 0. Indicates the starting epoch. Usually, if we are resuming from a
        checkpoint saved at a previous training phase at epoch N, then we should specify
        this value as N+1.
    :$num_epoch : Int
        Number of epochs for the training.
=cut


method fit(
    AI::MXNet::DataIter                 $train_data,
    Maybe[AI::MXNet::DataIter]         :$eval_data=,
    EvalMetric                         :$eval_metric='acc',
    Maybe[Callback]|ArrayRef[Callback] :$epoch_end_callback=,
    Maybe[Callback]|ArrayRef[Callback] :$batch_end_callback=,
    Str                                :$kvstore='local',
    Optimizer                          :$optimizer='sgd',
    HashRef                            :$optimizer_params={ learning_rate => 0.01 },
    Maybe[Callback]|ArrayRef[Callback] :$eval_end_callback=,
    Maybe[Callback]|ArrayRef[Callback] :$eval_batch_end_callback=,
    AI::MXNet::Initializer             :$initializer=AI::MXNet::Initializer->Uniform(scale => 0.01),
    Maybe[HashRef[AI::MXNet::NDArray]] :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] :$aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_rebind=0,
    Bool                               :$force_init=0,
    Int                                :$begin_epoch=0,
    Int                                :$num_epoch,
    Maybe[EvalMetric]                  :$validation_metric=,
    Maybe[AI::MXNet::Monitor]          :$monitor=
)
{
    $self->bind(
        data_shapes  => $train_data->provide_data,
        label_shapes => $train_data->provide_label,
        for_training => 1,
        force_rebind => $force_rebind
    );
    if($monitor)
    {
        $self->install_monitor($monitor);
    }
    $self->init_params(
        initializer   => $initializer,
        arg_params    => $arg_params,
        aux_params    => $aux_params,
        allow_missing => $allow_missing,
        force_init    => $force_init
    );
    $self->init_optimizer(
        kvstore          => $kvstore,
        optimizer        => $optimizer,
        optimizer_params => $optimizer_params
    );

    if(not defined $validation_metric)
    {
        $validation_metric = $eval_metric;
    }
    $eval_metric = AI::MXNet::Metric->create($eval_metric)
        unless blessed $eval_metric;

    ################################################################################
    # training loop
    ################################################################################
    for my $epoch ($begin_epoch..$num_epoch-1)
    {
        my $tic = time;
        $eval_metric->reset;
        my $nbatch = 0;
        my $end_of_batch = 0;
        my $next_data_batch = <$train_data>;
        while(not $end_of_batch)
        {
            my $data_batch = $next_data_batch;
            $monitor->tic if $monitor;
            $self->forward_backward($data_batch);
            $self->update;
            $next_data_batch = <$train_data>;
            if(defined $next_data_batch)
            {
                $self->prepare($next_data_batch);
            }
            else
            {
                $end_of_batch = 1;
            }
            $self->update_metric($eval_metric, $data_batch->label);
            $monitor->toc_print if $monitor;
            if(defined $batch_end_callback)
            {
                my $batch_end_params = AI::MXNet::BatchEndParam->new(
                    epoch       => $epoch,
                    nbatch      => $nbatch,
                    eval_metric => $eval_metric
                );
                for my $callback (@{ _as_list($batch_end_callback) })
                {
                    &{$callback}($batch_end_params);
                }
            }
            $nbatch++;
        }
        # one epoch of training is finished
        my $name_value = $eval_metric->get_name_value;
        while(my ($name, $val) = each %{ $name_value })
        {
            $self->logger->info('Epoch[%d] Train-%s=%f', $epoch, $name, $val);
        }
        my $toc = time;
        $self->logger->info('Epoch[%d] Time cost=%.3f', $epoch, ($toc-$tic));

        # sync aux params across devices
        my ($arg_params, $aux_params) = $self->get_params;
        $self->set_params($arg_params, $aux_params);

        if($epoch_end_callback)
        {
            for my $callback (@{ _as_list($epoch_end_callback) })
            {
                &{$callback}($epoch, $self->get_symbol, $arg_params, $aux_params);
            }
        }
        #----------------------------------------
        # evaluation on validation set
        if(defined $eval_data)
        {
            my $res = $self->score(
                $eval_data,
                $validation_metric,
                score_end_callback => $eval_end_callback,
                batch_end_callback => $eval_batch_end_callback,
                epoch              => $epoch
            );
            #TODO: pull this into default
            while(my ($name, $val) = each %{ $res })
            {
                $self->logger->info('Epoch[%d] Validation-%s=%f', $epoch, $name, $val);
            }
        }
        # end of 1 epoch, reset the data-iter for another epoch
        $train_data->reset;
    }
}

################################################################################
# Symbol information
################################################################################

=head2 get_symbol

    The symbol used by this module.
=cut
method get_symbol() { $self->symbol }

=head2 data_names

    An array ref of names for data required by this module.
=cut
method data_names() { confess("NotImplemented") }

=head2 output_names

    An array ref of names for the outputs of this module.
=cut
method output_names() { confess("NotImplemented") }

################################################################################
# Input/Output information
################################################################################

=head2 data_shapes

    An array ref of AI::MXNet::DataDesc objects specifying the data inputs to this module.
=cut
method data_shapes() { confess("NotImplemented") }

=head2 label_shapes

    A array ref of AI::MXNet::DataDesc objects specifying the label inputs to this module.
    If this module does not accept labels -- either it is a module without a loss
    function, or it is not binded for training, then this should return an empty
    array ref.
=cut
method label_shapes() { confess("NotImplemented") }

=head2 output_shapes

    An array ref of (name, shape) array refs specifying the outputs of this module.
=cut
method output_shapes() { confess("NotImplemented") }

################################################################################
# Parameters of a module
################################################################################

=head2 get_params

    The parameters, these are potentially a copies of the the actual parameters used
    to do computation on the device.

    Returns
    -------
    ($arg_params, $aux_params), a pair of hash refs of name to value mapping.
=cut

method get_params() { confess("NotImplemented") }

=head2 init_params

    Initialize the parameters and auxiliary states.

    Parameters
    ----------
    :$initializer : Maybe[AI::MXNet::Initializer]
        Called to initialize parameters if needed.
    :$arg_params= : Maybe[HashRef[AI::MXNet::NDArray]]
        If not undef, should be a hash ref of existing arg_params.
    :$aux_params : Maybe[HashRef[AI::MXNet::NDArray]]
        If not undef, should be a hash ref of existing aux_params.
    :$allow_missing=0 : Bool
        If true, params could contain missing values, and the initializer will be
        called to fill those missing params.
    :$force_init=0 : Bool
        If true, will force re-initialize even if already initialized.
=cut

method init_params(
    Maybe[AI::MXNet::Initializer]      :$initializer=AI::MXNet::Initializer->Uniform(0.01),
    Maybe[HashRef[AI::MXNet::NDArray]] :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] :$aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_init=0
)
{
    confess("NotImplemented");
}

=head2 set_params

    Assign parameter and aux state values.

    Parameters
    ----------
    $arg_params= : Maybe[HashRef[AI::MXNet::NDArray]]
        Hash ref of name to value (NDArray) mapping.
    $aux_params= : Maybe[HashRef[AI::MXNet::NDArray]]
        Hash Ref of name to value (`NDArray`) mapping.
    :$allow_missing=0 : Bool
        If true, params could contain missing values, and the initializer will be
        called to fill those missing params.
    :$force_init=0 : Bool
        If true, will force re-initialize even if already initialized.
=cut

method set_params(
    Maybe[HashRef[AI::MXNet::NDArray]]  $arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]]  $aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_init=0
)
{
    $self->init_params(
        initializer   => undef,
        arg_params    => $arg_params,
        aux_params    => $aux_params,
        allow_missing => $allow_missing,
        force_init    => $force_init
    );
}

=head2 save_params

    Save model parameters to file.

    Parameters
    ----------
    $fname : str
        Path to output param file.
    $arg_params= : Maybe[HashRef[AI::MXNet::NDArray]]
    $aux_params= : Maybe[HashRef[AI::MXNet::NDArray]]
=cut

method save_params(
    Str $fname,
    Maybe[HashRef[AI::MXNet::NDArray]] $arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] $aux_params=
)
{
    ($arg_params, $aux_params) = $self->get_params
        unless (defined $arg_params and defined $aux_params);
    my %save_dict;
    while(my ($k, $v) = each %{ $arg_params })
    {
        $save_dict{"arg:$k"} = $v->as_in_context(AI::MXNet::Context->cpu);
    }
    while(my ($k, $v) = each %{ $aux_params })
    {
        $save_dict{"aux:$k"} = $v->as_in_context(AI::MXNet::Context->cpu);
    }
    AI::MXNet::NDArray->save($fname, \%save_dict);
}

=head2 load_params

    Load model parameters from file.

    Parameters
    ----------
    $fname : str
        Path to input param file.
=cut

method load_params(Str $fname)
{
    my %save_dict = %{ AI::MXNet::NDArray->load($fname) };
    my %arg_params;
    my %aux_params;
    while(my ($k, $v) = each %save_dict)
    {
        my ($arg_type, $name) = split(/:/, $k, 2);
        if($arg_type eq 'arg')
        {
            $arg_params{ $name } = $v;
        }
        elsif($arg_type eq 'aux')
        {
            $aux_params{ $name } = $v;
        }
        else
        {
            confess("Invalid param file $fname");
        }
    }
    $self->set_params(\%arg_params, \%aux_params);
}

=head2 get_states

    The states from all devices

    Parameters
    ----------
    $merge_multi_context=1 : Bool
        Default is true (1). In the case when data-parallelism is used, the states
        will be collected from multiple devices. A true value indicate that we
        should merge the collected results so that they look like from a single
        executor.

    Returns
    -------
    If $merge_multi_context is 1, it is like [$out1, $out2]. Otherwise, it
    is like [[$out1_dev1, $out1_dev2], [$out2_dev1, $out2_dev2]]. All the output
    elements are AI::MXNet::NDArray.
=cut

method get_states(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    assert(not $merge_multi_context);
    return [];
}

=head2 set_states

    Set value for states. You can specify either $states or $value, not both.

    Parameters
    ----------
    $states= : Maybe[ArrayRef[ArrayRef[AI::MXNet::NDArray]]]
        source states arrays formatted like [[$state1_dev1, $state1_dev2],
            [$state2_dev1, $state2_dev2]].
    $value= : Maybe[Num]
        a single scalar value for all state arrays.
=cut

method set_states(Maybe[ArrayRef[ArrayRef[AI::MXNet::NDArray]]] $states=, Maybe[Num] $value=)
{
    assert($self->binded and $self->params_initialized);
    assert(not $states and not $value);
}


=head2 install_monitor

    Install monitor on all executors

    Parameters
    ----------
    $mon : AI::MXNet::Monitor
=cut

method install_monitor(AI::MXNet::Monitor $mon) { confess("NotImplemented") }

=head2 prepare

    Prepare the module for processing a data batch.

    Usually involves switching a bucket and reshaping.

    Parameters
    ----------
    $data_batch : AI::MXNet::DataBatch
=cut

method prepare(AI::MXNet::DataBatch $data_batch){}

################################################################################
# Computations
################################################################################

=head2 forward

    Forward computation.

    Parameters
    ----------
    $data_batch : DataBatch
        Could be anything with similar API implemented.
    :$is_train= : Bool
        Default is undef, which means is_train takes the value of $self->for_training.
=cut

method forward(AI::MXNet::DataBatch $data_batch, Bool :$is_train=) { confess("NotImplemented") }

=head2 backward

    Backward computation.

    Parameters
    ----------
    $out_grads : Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]], optional
        Gradient on the outputs to be propagated back.
        This parameter is only needed when bind is called
        on outputs that are not a loss function.
=cut

method backward(Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]] $out_grads=)
{
    confess("NotImplemented")
}

=head2 get_outputs

    The outputs of the previous forward computation.

    Parameters
    ----------
    $merge_multi_context=1 : Bool
=cut

method get_outputs(Bool $merge_multi_context=1) { confess("NotImplemented") }

=head2 get_input_grads

    The gradients to the inputs, computed in the previous backward computation.

    Parameters
    ----------
    $merge_multi_context=1 : Bool
=cut

method get_input_grads(Bool $merge_multi_context=1) { confess("NotImplemented") }

=head2 update

    Update parameters according to the installed optimizer and the gradients computed
    in the previous forward-backward batch.
=cut

method update() { confess("NotImplemented") }

=head2 update_metric

    Evaluate and accumulate evaluation metric on outputs of the last forward computation.

    Parameters
    ----------
    $eval_metric : EvalMetric
    $labels : ArrayRef[AI::MXNet::NDArray]
        Typically $data_batch->label.
=cut

method update_metric(EvalMetric $eval_metric, ArrayRef[AI::MXNet::NDArray] $labels)
{
    confess("NotImplemented")
}

################################################################################
# module setup
################################################################################

=head2 bind

    Binds the symbols in order to construct the executors. This is necessary
    before the computations can be performed.

    Parameters
    ----------
    $data_shapes : ArrayRef[AI::MXNet::DataDesc]
        Typically is $data_iter->provide_data.
    :$label_shapes= : Maybe[ArrayRef[AI::MXNet::DataDesc]]
        Typically is $data_iter->provide_label.
    :$for_training=1 : Bool
        Default is 1. Whether the executors should be bind for training.
    :$inputs_need_grad=0 : Bool
        Default is 0. Whether the gradients to the input data need to be computed.
        Typically this is not needed. But this might be needed when implementing composition
        of modules.
    :$force_rebind=0 : Bool
        Default is 0. This function does nothing if the executors are already
        binded. But with this as 1, the executors will be forced to rebind.
    :$shared_module= : A subclass of AI::MXNet::Module::Base
        Default is undef. This is used in bucketing. When not undef, the shared module
        essentially corresponds to a different bucket -- a module with different symbol
        but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
    :$grad_req='write' : Str|ArrayRef[Str]|HashRef[Str]
        Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
        (defaults to 'write').
        Can be specified globally (str) or for each argument (array ref, hash ref).
=cut

method bind(
    ArrayRef[AI::MXNet::DataDesc]         $data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc]] :$label_shapes=,
    Bool                                 :$for_training=1,
    Bool                                 :$inputs_need_grad=0,
    Bool                                 :$force_rebind=0,
    Maybe[AI::MXNet::BaseModule]         :$shared_module=,
    Str|ArrayRef[Str]|HashRef[Str]       :$grad_req='write'
)
{
    confess("NotImplemented")
}

=head2 init_optimizer

    Install and initialize optimizers.

    Parameters
    ----------
    :$kvstore='local' : str or KVStore
    :$optimizer='sgd' : str or Optimizer
    :$optimizer_params={ learning_rate => 0.01 } : hash ref
    :$force_init=0 : Bool
=cut

method init_optimizer(
    Str        :$kvstore='local',
    Optimizer  :$optimizer='sgd',
    HashRef    :$optimizer_params={ learning_rate => 0.01 },
    Bool       :$force_init=0
)
{
    confess("NotImplemented")
}

################################################################################
# misc
################################################################################

=head2 symbol

    The symbol associated with this module.

    Except for AI::MXNet::Module, for other types of modules (e.g. AI::MXNet::Module::Bucketing), this
    property might not be a constant throughout its life time. Some modules might
    not even be associated with any symbols.
=cut

method symbol()
{
    return $self->_symbol;
}

1;
