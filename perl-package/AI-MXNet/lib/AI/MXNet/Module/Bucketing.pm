package AI::MXNet::Module::Bucketing;
use Mouse;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;

=encoding UTF-8

=head1 NAME

AI::MXNet::Module::Bucketing

=head1 SYNOPSIS

    my $buckets = [10, 20, 30, 40, 50, 60];
    my $start_label   = 1;
    my $invalid_label = 0;

    my ($train_sentences, $vocabulary) = tokenize_text(
        './data/ptb.train.txt', start_label => $start_label,
        invalid_label => $invalid_label
    );
    my ($validation_sentences) = tokenize_text(
        './data/ptb.test.txt', vocab => $vocabulary,
        start_label => $start_label, invalid_label => $invalid_label
    );
    my $data_train  = mx->rnn->BucketSentenceIter(
        $train_sentences, $batch_size, buckets => $buckets,
        invalid_label => $invalid_label
    );
    my $data_val    = mx->rnn->BucketSentenceIter(
        $validation_sentences, $batch_size, buckets => $buckets,
        invalid_label => $invalid_label
    );

    my $stack = mx->rnn->SequentialRNNCell();
    for my $i (0..$num_layers-1)
    {
        $stack->add(mx->rnn->LSTMCell(num_hidden => $num_hidden, prefix => "lstm_l${i}_"));
    }

    my $sym_gen = sub {
        my $seq_len = shift;
        my $data  = mx->sym->Variable('data');
        my $label = mx->sym->Variable('softmax_label');
        my $embed = mx->sym->Embedding(
            data => $data, input_dim => scalar(keys %$vocabulary),
            output_dim => $num_embed, name => 'embed'
        );
        $stack->reset;
        my ($outputs, $states) = $stack->unroll($seq_len, inputs => $embed, merge_outputs => 1);
        my $pred = mx->sym->Reshape($outputs, shape => [-1, $num_hidden]);
        $pred    = mx->sym->FullyConnected(data => $pred, num_hidden => scalar(keys %$vocabulary), name => 'pred');
        $label   = mx->sym->Reshape($label, shape => [-1]);
        $pred    = mx->sym->SoftmaxOutput(data => $pred, label => $label, name => 'softmax');
        return ($pred, ['data'], ['softmax_label']);
    };

    my $contexts;
    if(defined $gpus)
    {
        $contexts = [map { mx->gpu($_) } split(/,/, $gpus)];
    }
    else
    {
        $contexts = mx->cpu(0);
    }

    my $model = mx->mod->BucketingModule(
        sym_gen             => $sym_gen,
        default_bucket_key  => $data_train->default_bucket_key,
        context             => $contexts
    );

    $model->fit(
        $data_train,
        eval_data           => $data_val,
        eval_metric         => mx->metric->Perplexity($invalid_label),
        kvstore             => $kv_store,
        optimizer           => $optimizer,
        optimizer_params    => {
                                    learning_rate => $lr,
                                    momentum      => $mom,
                                    wd            => $wd,
                            },
        initializer         => mx->init->Xavier(factor_type => "in", magnitude => 2.34),
        num_epoch           => $num_epoch,
        batch_end_callback  => mx->callback->Speedometer($batch_size, $disp_batches),
        ($chkp_epoch ? (epoch_end_callback  => mx->rnn->do_rnn_checkpoint($stack, $chkp_prefix, $chkp_epoch)) : ())
    );

=head1 DESCRIPTION

    Implements the AI::MXNet::Module::Base API, and allows multiple
    symbols to be used depending on the `bucket_key` provided by each different
    mini-batch of data
=cut


=head2 new

    Parameters
    ----------
    $sym_gen : subref or any perl object that overloads &{} op
        A sub when called with a bucket key, returns a list with triple
        of ($symbol, $data_names, $label_names).
    $default_bucket_key : str or anything else
        The key for the default bucket.
    $logger : Logger
    $context : AI::MXNet::Context or array ref of AI::MXNet::Context objects
        Default is cpu(0)
    $work_load_list : array ref of Num
        Default is undef, indicating uniform workload.
    $fixed_param_names: arrayref of str
        Default is undef, indicating no network parameters are fixed.
    $state_names : arrayref of str
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
has '_curr_bucket_key'    => (is => 'rw', init_arg => undef);
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
    $self->_curr_bucket_key(undef);
}

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

method data_shapes()
{
    assert($self->binded);
    return $self->_curr_module->data_shapes;
}

method label_shapes()
{
    assert($self->binded);
    return $self->_curr_module->label_shapes;
}

method output_shapes()
{
    assert($self->binded);
    return $self->_curr_module->output_shapes;
}

method get_params()
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->_p->_params_dirty($self->_params_dirty);
    my ($arg_params, $aux_params) = $self->_curr_module->get_params;
    $self->_params_dirty(0);
    return ($arg_params, $aux_params);
}

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

method init_params(
    AI::MXNet::Initializer             :$initializer=AI::MXNet::Initializer->Uniform(scale => 0.01),
    Maybe[HashRef[AI::MXNet::NDArray]] :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]] :$aux_params=,
    Bool                               :$allow_missing=0,
    Bool                               :$force_init=0
)
{
    return if($self->params_initialized and not $force_init);
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

method set_states(:$states=, :$value=)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->set_states(states => $states, value => $value);
}

=head2 bind

    Binding for a AI::MXNet::Module::Bucketing means setting up the buckets and bind the
    executor for the default bucket key. Executors corresponding to other keys are
    binded afterwards with switch_bucket.

    Parameters
    ----------
    :$data_shapes : ArrayRef[AI::MXNet::DataDesc|NameShape]
        This should correspond to the symbol for the default bucket.
    :$label_shapes= : Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
        This should correspond to the symbol for the default bucket.
    :$for_training : Bool
        Default is 1.
    :$inputs_need_grad : Bool
        Default is 0.
    :$force_rebind : Bool
        Default is 0.
    :$shared_module : AI::MXNet::Module::Bucketing
        Default is undef. This value is currently not used.
    :$grad_req : str, array ref of str, hash ref of str to str
        Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
        (defaults to 'write').
        Can be specified globally (str) or for each argument (array ref, hash ref).
    :$bucket_key : str
        bucket key for binding. by default is to use the ->default_bucket_key
=cut

method bind(
    ArrayRef[AI::MXNet::DataDesc|NameShape]                   :$data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]            :$label_shapes=,
    Bool                                                      :$for_training=1,
    Bool                                                      :$inputs_need_grad=0,
    Bool                                                      :$force_rebind=0,
    Maybe[AI::MXNet::BaseModule]                              :$shared_module=,
    Str|ArrayRef[Str]|HashRef[Str]                            :$grad_req='write',
    Maybe[Str]                                                :$bucket_key=
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
    $self->_curr_bucket_key($self->_default_bucket_key);
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
    :$bucket_key : str (or any perl object that overloads "" op)
        The key of the target bucket.
    :$data_shapes :  Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
        Typically $data_batch->provide_data.
    :$label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]
        Typically $data_batch->provide_label.
=cut

method switch_bucket(
    Maybe[ArrayRef[AI::MXNet::DataDesc|NameShape]]            :$data_shapes=,
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
        $self->_buckets->{ $bucket_key } = $module;
    }
    $self->_curr_module($self->_buckets->{ $bucket_key });
    $self->_curr_bucket_key($bucket_key);
}

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

method prepare(AI::MXNet::DataBatch $data_batch)
{
    assert($self->binded and $self->params_initialized);
    ## perform bind if have not done so yet
    my $original_bucket_key = $self->_curr_bucket_key;
    $self->switch_bucket(
        bucket_key   => $data_batch->bucket_key,
        data_shapes  => $data_batch->provide_data,
        label_shapes => $data_batch->provide_label
    );
    # switch back
    $self->switch_bucket(bucket_key => $original_bucket_key);
}

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

method backward(Maybe[ArrayRef[AI::MXNet::NDArray]|AI::MXNet::NDArray] $out_grads=)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->backward($out_grads);
}

method update()
{
    assert($self->binded and $self->params_initialized and $self->optimizer_initialized);
    $self->_params_dirty(1);
    $self->_curr_module->update;
}

method get_outputs(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized);
    return $self->_curr_module->get_outputs($merge_multi_context);
}

method get_input_grads(Bool $merge_multi_context=1)
{
    assert($self->binded and $self->params_initialized and $self->inputs_need_grad);
    return $self->_curr_module->get_input_grads($merge_multi_context);
}

method update_metric(
    AI::MXNet::EvalMetric $eval_metric,
    ArrayRef[AI::MXNet::NDArray] $labels
)
{
    assert($self->binded and $self->params_initialized);
    $self->_curr_module->update_metric($eval_metric, $labels);
}

method symbol()
{
    assert($self->binded);
    return $self->_curr_module->symbol;
}

method get_symbol()
{
    assert($self->binded);
    return $self->_buckets->{ $self->_default_bucket_key }->symbol;
}

method install_monitor(AI::MXNet::Monitor $mon)
{
    assert($self->binded);
    for my $mod (values %{ $self->_buckets })
    {
        $mod->install_monitor($mon);
    }
}

1;
