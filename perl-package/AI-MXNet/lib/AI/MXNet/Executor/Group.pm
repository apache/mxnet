package AI::MXNet::Executor::Group;
use strict;
use warnings;
use Scalar::Util qw(blessed);
use List::Util qw(sum min);
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Executor::Group - Manager for a group of executors working in different contexts.
=cut

func _split_input_slice($batch_size, $work_load_list)
{
    my $total_work_load = sum(@{ $work_load_list });
    my @batch_num_list = map { # perl does not have builtin round
        int(($_ * $batch_size / $total_work_load) + 0.5)
    } @{ $work_load_list };
    my $batch_num_sum = sum(@batch_num_list);
    my @slices;
    if($batch_num_sum < $batch_size)
    {
        $batch_num_list[-1] += $batch_size - $batch_num_sum;
    }
    my $end = 0;
    for my $batch_num (@batch_num_list)
    {
        my $begin = int(min($end, $batch_size));
        $end = int(min($begin + $batch_num, $batch_size));
        if($begin >= $end)
        {
            confess('Too many slices such that some splits are empty');
        }
        push @slices, [$begin, $end];
    }
    return \@slices;
}
# Load a array ref of arrays into a array ref of arrays specified by slices
func _load_general($data, $targets, $major_axis)
{
    zip(sub {
        my ($d_src, $d_targets, $axis) = @_;
        if(blessed($d_targets) and $d_targets->isa('AI::MXNet::NDarray'))
        {
            $d_src->copyto($d_targets);
        }
        elsif(ref $d_targets eq 'ARRAY' and blessed $d_targets->[0])
        {
            zip(sub {
                my ($src, $dst) = @_;
                $src->copyto($dst);
            }, $d_src, $d_targets);
        }
        else
        {
            for my $d (@{ $d_targets })
            {
                my ($slice_idx, $d_dst) = @{ $d };
                if($axis >= 0)
                {
                    # copy slice
                    my $end   = $d_src->shape;
                    my $begin = [(0) x @{ $end }];
                    $begin->[$axis] = $slice_idx->[0];
                    $end->[$axis]   = $slice_idx->[1];
                    if($d_src->context == $d_dst->context)
                    {
                        $d_src->crop({ begin => $begin, end => $end, out => $d_dst });
                    }
                    else
                    {
                        # on different device, crop and then do cross device copy
                        my $d_dst_copy = $d_src->crop({ begin => $begin, end => $end });
                        $d_dst_copy->copyto($d_dst);
                    }
                }
                else
                {
                    $d_src->copyto($d_dst);
                }
            }
        }
    }, $data, $targets, $major_axis);
}

# Load data into sliced arrays
func _load_data($batch, $targets, $major_axis)
{
    _load_general($batch->data, $targets, $major_axis);
}

# Load label into sliced arrays
func _load_label($batch, $targets, $major_axis)
{
    _load_general($batch->label, $targets, $major_axis);
}

# Merge outputs that live on multiple context into one, so that they look
# like living on one context.
func _merge_multi_context($outputs, $major_axis)
{
    my @rets;
    zip(sub {
        my ($tensors, $axis) = @_;
        if($axis >= 0)
        {
            if(@$tensors == 1)
            {
                push @rets, $tensors->[0];
            }
            else
            {
                my $ctx = $tensors->[0]->context;
                push @rets, AI::MXNet::NDArray->concat((map { $_->as_in_context($ctx) } @$tensors), { dim => $axis });
            }
        }
        else
        {
            # negative axis means the there is no batch_size axis, and all the
            # results should be the same on each device. We simply take the
            # first one, without checking they are actually the same
            push @rets, $tensors->[0];
        }
    }, $outputs, $major_axis);
    return \@rets;
}

## TODO
## this class is here because of https://github.com/gfx/p5-Mouse/pull/67
## once 2.4.7 version of Mouse in Ubuntu for affected Perl version
## these accessors should be merged into main class
package AI::MXNet::DataParallelExecutorGroup::_private;
use Mouse;
has [qw/output_layouts label_layouts arg_names aux_names
        batch_size slices execs data_arrays
        label_arrays param_arrays grad_arrays aux_arrays
        data_layouts shared_data_arrays input_grad_arrays
        _default_execs state_arrays/
    ] => (is => 'rw', init_arg => undef);

package AI::MXNet::DataParallelExecutorGroup;
use Mouse;
use AI::MXNet::Base;
use List::Util qw(sum);

=head1 DESCRIPTION

    DataParallelExecutorGroup is a group of executors that lives on a group of devices.
    This is a helper class used to implement data parallelization. Each mini-batch will
    be split and run on the devices.

    Parameters for constructor
    ----------
    symbol : AI::MXNet::Symbol
        The common symbolic computation graph for all executors.
    contexts : ArrayRef[AI::MXNet::Context]
        A array ref of contexts.
    workload : ArrayRef[Num]
        If not undef, could be an array ref of numbers that specify the workload to be assigned
        to different context. Larger number indicate heavier workload.
    data_shapes : ArrayRef[NameShape|AI::MXNet::DataDesc]
        Should be a array ref of [name, shape] array refs, for the shapes of data. Note the order is
        important and should be the same as the order that the `DataIter` provide the data.
    label_shapes : Maybe[ArrayRef[NameShape|AI::MXNet::DataDesc]]
        Should be a array ref of [$name, $shape] array refs, for the shapes of label. Note the order is
        important and should be the same as the order that the `DataIter` provide the label.
    param_names : ArrayRef[Str]
        A array ref of strings, indicating the names of parameters (e.g. weights, filters, etc.)
        in the computation graph.
    for_training : Bool
        Indicate whether the executors should be bind for training. When not doing training,
        the memory for gradients will not be allocated.
    inputs_need_grad : Bool
        Indicate whether the gradients for the input data should be computed. This is currently
        not used. It will be useful for implementing composition of modules.
    shared_group : AI::MXNet::DataParallelExecutorGroup
        Default is undef. This is used in bucketing. When not undef, it should be a executor
        group corresponding to a different bucket. In other words, it will correspond to a different
        symbol but with the same set of parameters (e.g. unrolled RNNs with different lengths).
        In this case, many memory will be shared.
    logger : Logger
        Default is AI::MXNet::Logging->get_logger.
    fixed_param_names: Maybe[ArrayRef[Str]]
        Indicate parameters to be fixed during training. Parameters in this array ref will not allocate
        space for gradient, nor do gradient calculation.
    grad_req : ArrayRef[GradReq]|HashRef[GradReq]|GradReq
        Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
        (default to 'write').
        Can be specified globally (str) or for each argument (array ref, hash ref).
    state_names: Maybe[ArrayRef[Str]]
=cut

has 'symbol'            => (is => 'ro', isa => 'AI::MXNet::Symbol', required => 1);
has 'contexts'          => (is => 'ro', isa => 'ArrayRef[AI::MXNet::Context]', required => 1);
has 'workload'          => (is => 'ro', isa => 'ArrayRef[Num]', default => sub { [] });
has 'data_shapes'       => (is => 'rw', isa => 'ArrayRef[NameShape|AI::MXNet::DataDesc]', required => 1);
has 'label_shapes'      => (is => 'rw', isa => 'Maybe[ArrayRef[NameShape|AI::MXNet::DataDesc]]');
has 'param_names'       => (is => 'ro', isa => 'ArrayRef[Str]', required => 1);
has 'for_training'      => (is => 'ro', isa => 'Bool', required => 1);
has 'inputs_need_grad'  => (is => 'ro', isa => 'Bool', default  => 0);
has 'shared_group'      => (is => 'ro', isa => 'Maybe[AI::MXNet::DataParallelExecutorGroup]');
has 'logger'            => (is => 'ro', default => sub { AI::MXNet::Logging->get_logger });
has 'fixed_param_names' => (is => 'rw', isa => 'Maybe[ArrayRef[Str]]');
has 'state_names'       => (is => 'rw', isa => 'Maybe[ArrayRef[Str]]');
has 'grad_req'          => (is => 'rw', isa => 'ArrayRef[GradReq]|HashRef[GradReq]|GradReq', default=>'write');
has '_p'                => (is => 'rw', init_arg => undef);
sub BUILD
{
    my $self = shift;
    my $p = AI::MXNet::DataParallelExecutorGroup::_private->new;
    $p->arg_names($self->symbol->list_arguments);
    $p->aux_names($self->symbol->list_auxiliary_states);
    $p->execs([]);
    $self->_p($p);
    $self->grad_req('null') if not $self->for_training;
    $self->fixed_param_names([]) unless defined $self->fixed_param_names;
    $self->state_names([]) unless defined $self->state_names;
    my $data_shapes = [];
    for my $d (@{ $self->data_shapes })
    {
        $d = AI::MXNet::DataDesc->new(name => $d->[0], shape => $d->[1])
            unless blessed $d;
        push @{ $data_shapes }, $d;
    }
    $self->data_shapes($data_shapes);
    if(defined $self->label_shapes)
    {
        my $label_shapes = [];
        for my $l (@{ $self->label_shapes })
        {
            $l = AI::MXNet::DataDesc->new(name => $l->[0], shape => $l->[1])
                unless blessed $l;
            push @{ $label_shapes }, $l;
        }
        $self->label_shapes($label_shapes);
    }
    my %data_names  = map { $_->name => 1 } @{ $self->data_shapes };
    my %param_names = map { $_    =>    1 } @{ $self->param_names };
    my %fixed_param_names = map { $_ => 1 } @{ $self->fixed_param_names };
    my %grad_req;
    if(not ref $self->grad_req)
    {
        for my $k (@{ $self->_p->arg_names })
        {
            if(exists $param_names{ $k })
            {
                $grad_req{$k} = exists $fixed_param_names{ $k } ? 'null' : $self->grad_req;
            }
            elsif(exists $data_names{ $k })
            {
                $grad_req{$k} = $self->inputs_need_grad ? $self->grad_req : 'null';
            }
            else
            {
                $grad_req{$k} = 'null';
            }
        }
    }
    elsif(ref $self->grad_req eq 'ARRAY')
    {
        @grad_req{ @{ $self->_p->arg_names } } = @{ $self->grad_req };
    }
    else
    {
        for my $k (@{ $self->_p->arg_names })
        {
            if(exists $param_names{ $k })
            {
                $grad_req{$k} = exists $fixed_param_names{ $k } ? 'null' : 'write';
            }
            elsif(exists $data_names{ $k })
            {
                $grad_req{$k} = $self->inputs_need_grad ? 'write' : 'null';
            }
            else
            {
                $grad_req{$k} = 'null';
            }
        }
        %grad_req = (%grad_req, %{ $self->grad_req });
    }
    $self->grad_req(\%grad_req);
    if(defined $self->shared_group)
    {
        $self->_p->shared_data_arrays($self->shared_group->_p->shared_data_arrays);
    }
    else
    {
        $self->_p->shared_data_arrays([map { +{} } 0..@{ $self->contexts }-1]);
    }
    $self->_p->output_layouts([
        map {
            AI::MXNet::DataDesc->get_batch_axis($self->symbol->slice($_)->attr('__layout__'))
        } @{ $self->symbol->list_outputs }
    ]);
    $self->bind_exec($self->data_shapes, $self->label_shapes, $self->shared_group);
}

=decide_slices

    Decide the slices for each context according to the workload.

    Parameters
    ----------
    $data_shapes : ArrayRef[AI::MXNet::DataDesc]
=cut

method decide_slices(ArrayRef[AI::MXNet::DataDesc] $data_shapes)
{
    confess("empty data_shapes array") unless @{ $data_shapes } > 0;
    my $major_axis = [map { AI::MXNet::DataDesc->get_batch_axis($_->layout) } @{ $data_shapes }];
    zip(sub {
        my ($desc, $axis) = @_;
        return if($axis == -1);
        my $batch_size = $desc->shape->[$axis];
        if(defined $self->_p->batch_size)
        {
            confess(
                "all data must have the same batch size: "
                . sprintf("batch_size = %d, but ", $self->_p->batch_size)
                . sprintf("%s has shape %s", $desc->name, '('. join(',', @{ $desc->shape }) . ')')
            ) unless $batch_size == $self->_p->batch_size;
        }
        else
        {
            $self->_p->batch_size($batch_size);
            $self->_p->slices(AI::MXNet::Executor::Group::_split_input_slice($self->_p->batch_size, $self->workload));
        }
    }, $data_shapes, $major_axis);
    return $major_axis;
}

# Collect internal arrays from executors.
method _collect_arrays()
{
    # convenient data structures
    $self->_p->data_arrays([]);
    for my $d (@{ $self->data_shapes })
    {
        my $name = $d->name;
        my @tmp;
        for my $i (0..@{ $self->_p->execs }-1)
        {
            push @tmp, [ $self->_p->slices->[$i], $self->_p->execs->[$i]->arg_dict->{$name} ];
        }
        push @{ $self->_p->data_arrays }, \@tmp;
    }
    if(defined $self->label_shapes)
    {
        $self->_p->label_arrays([]);
        for my $l (@{ $self->label_shapes })
        {
            my $name = $l->name;
            my @tmp;
            for my $i (0..@{ $self->_p->execs }-1)
            {
                push @tmp, [ $self->_p->slices->[$i], $self->_p->execs->[$i]->arg_dict->{$name} ];
            }
            push @{ $self->_p->label_arrays }, \@tmp;
        }
    }
    $self->_p->param_arrays([]);
    my %param_names = map { $_ => 1 } @{ $self->param_names };
    for my $i (0..@{ $self->_p->arg_names }-1)
    {
        my $name = $self->_p->arg_names->[$i];
        if(exists $param_names{$name})
        {
            my @tmp;
            for my $exec (@{ $self->_p->execs })
            {
                push @tmp, $exec->arg_arrays->[$i];
            }
            push @{ $self->_p->param_arrays }, \@tmp;
        }
    }
    $self->_p->state_arrays([]);
    for my $i (0..@{ $self->state_names }-1)
    {
        my $name = $self->state_names->[$i];
        my @tmp;
        for my $exec (@{ $self->_p->execs })
        {
            push @tmp, $exec->arg_dict->{$name};
        }
        push @{ $self->_p->state_arrays }, \@tmp;
    }
    if($self->for_training)
    {
        $self->_p->grad_arrays([]);
        for my $i (0..@{ $self->_p->arg_names }-1)
        {
            my $name = $self->_p->arg_names->[$i];
            if(exists $param_names{$name})
            {
                my @tmp;
                for my $exec (@{ $self->_p->execs })
                {
                    push @tmp, $exec->grad_arrays->[$i];
                }
                push @{ $self->_p->grad_arrays }, \@tmp;
            }
        }
    }
    my @data_names = map { $_->name } @{ $self->data_shapes };
    my $j = 0; my %arg_names  = map { $_ => $j++ } @{ $self->_p->arg_names };
    if($self->inputs_need_grad)
    {
        $self->_p->input_grad_arrays([]);
        for my $name (@data_names)
        {
            next unless exists $arg_names{$name};
            my @tmp;
            for my $exec (@{ $self->_p->execs })
            {
                push @tmp, $exec->grad_arrays->[$arg_names{$name}];
            }
            push @{ $self->_p->input_grad_arrays }, \@tmp;
        }
    }
    $self->_p->aux_arrays([]);
    for my $i (0..@{ $self->_p->aux_names }-1)
    {
        my @tmp;
        for my $exec (@{ $self->_p->execs })
        {
            push @tmp, $exec->aux_arrays->[$i];
        }
        push @{ $self->_p->aux_arrays }, \@tmp;
    }
}

=head2 bind_exec

    Bind executors on their respective devices.

    Parameters
    ----------
    $data_shapes  : ArrayRef[AI::MXNet::DataDesc]
    $label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc]]
    $shared_group : Maybe[AI::MXNet::DataParallelExecutorGroup]
    $reshape      : Bool
=cut

method bind_exec(
    ArrayRef[AI::MXNet::DataDesc]               $data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc]]        $label_shapes=,
    Maybe[AI::MXNet::DataParallelExecutorGroup] $shared_group=,
    Bool                                        $reshape=0
)
{
    assert($reshape or not @{ $self->_p->execs });
    $self->_p->batch_size(undef);

    # calculate workload and bind executors
    $self->_p->data_layouts($self->decide_slices($data_shapes));
    # call it to make sure labels has the same batch size as data
    if(defined $label_shapes)
    {
        $self->_p->label_layouts($self->decide_slices($label_shapes));
    }

    for my $i (0..@{ $self->contexts }-1)
    {
        my $data_shapes_i = $self->_sliced_shape($data_shapes, $i, $self->_p->data_layouts);
        my $label_shapes_i = [];
        if(defined $label_shapes)
        {
            $label_shapes_i = $self->_sliced_shape($label_shapes, $i, $self->_p->label_layouts);
        }
        if($reshape)
        {
            my %combined_hash = map { $_->name => $_->shape } (@{ $data_shapes_i }, @{ $label_shapes_i });
            $self->_p->execs->[$i] = $self->_p->_default_execs->[$i]->reshape(
                \%combined_hash,
                allow_up_sizing => 1,
            );
        }
        else
        {
            push @{ $self->_p->execs }, $self->_bind_ith_exec($i, $data_shapes_i, $label_shapes_i, $shared_group);
        }
    }
    $self->data_shapes($data_shapes);
    $self->label_shapes($label_shapes);
    $self->_collect_arrays;
}

=head2 reshape

    Reshape executors.

    Parameters
    ----------
    $data_shapes : ArrayRef[AI::MXNet::DataDesc]
    $label_shapes : Maybe[ArrayRef[AI::MXNet::DataDesc]]
=cut


method reshape(
    ArrayRef[AI::MXNet::DataDesc]          $data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc]]   $label_shapes=
)
{
    return if($data_shapes eq $self->data_shapes and $label_shapes eq $self->label_shapes);
    if (not defined $self->_p->_default_execs)
    {
        $self->_p->_default_execs([@{ $self->_p->execs }]);
    }
    $self->bind_exec($data_shapes, $label_shapes, undef, 1);
}

=head2 set_params

    Assign, i.e. copy parameters to all the executors.

    Parameters
    ----------
    $arg_params : HashRef[AI::MXNet::NDArray]
        A dictionary of name to AI::MXNet::NDArray parameter mapping.
    $aux_params : HashRef[AI::MXNet::NDArray]
        A dictionary of name to AI::MXNet::NDArray auxiliary variable mapping.
=cut

method set_params(HashRef[AI::MXNet::NDArray] $arg_params, HashRef[AI::MXNet::NDArray] $aux_params)
{
    $_->copy_params_from($arg_params, $aux_params) for @{ $self->_p->execs };
}

=head2 get_params

    Copy data from each executor to arg_params and aux_params.

    Parameters
    ----------
    $arg_params : HashRef[AI::MXNet::NDArray]
        target parameter arrays
    $aux_params : HashRef[AI::MXNet::NDArray]
        target aux arrays

    Notes
    -----
    - This function will inplace update the NDArrays in arg_params and aux_params.
=cut

method get_params(HashRef[AI::MXNet::NDArray] $arg_params, HashRef[AI::MXNet::NDArray] $aux_params)
{
    my $weight = 0;
    zip(sub {
        my ($name, $block) = @_;
            my $weight = sum(map { $_->copyto(AI::MXNet::Context->cpu) } @{ $block }) / @{ $block };
            $weight->astype($arg_params->{$name}->dtype)->copyto($arg_params->{$name});
    }, $self->param_names, $self->_p->param_arrays);
    zip(sub {
        my ($name, $block) = @_;
            my $weight = sum(map { $_->copyto(AI::MXNet::Context->cpu) } @{ $block }) / @{ $block };
            $weight->astype($aux_params->{$name}->dtype)->copyto($aux_params->{$name});
    }, $self->_p->aux_names, $self->_p->aux_arrays);
}



method get_states($merge_multi_context=1)
{
    assert((not $merge_multi_context), "merge_multi_context=True is not supported for get_states yet.");
    return $self->_p->state_arrays;
}

method set_states($states, $value)
{
    if(defined $states)
    {
        assert((not defined $value), "Only one of states & value can be specified.");
        AI::MXNet::Executor::Group::_load_general($states, $self->_p->state_arrays, [(0)x@{ $states }]);
    }
    else
    {
        assert((defined $value), "At least one of states & value must be specified.");
        assert((not defined $states), "Only one of states & value can be specified.");
        for my $d_dst (@{ $self->_p->state_arrays })
        {
            for my $dst (@{ $d_dst })
            {
                $dst .= $value;
            }
        }
    }
}

=head2 forward

    Split the data_batch according to a workload and run forward on each devices.

    Parameters
    ----------
    data_batch : AI::MXNet::DataBatch
    Or could be any object implementing similar interface.

    is_train : bool
    The hint for the backend, indicating whether we are during training phase.
    Default is undef, then the value $self->for_training will be used.
=cut


method forward(AI::MXNet::DataBatch $data_batch, Maybe[Bool] $is_train=)
{
    AI::MXNet::Executor::Group::_load_data($data_batch, $self->_p->data_arrays, $self->_p->data_layouts);
    $is_train //= $self->for_training;
    if(defined $self->_p->label_arrays)
    {
        confess("assert not is_train or data_batch.label")
            unless (not $is_train or $data_batch->label);
        if($data_batch->label)
        {
            AI::MXNet::Executor::Group::_load_label($data_batch, $self->_p->label_arrays, $self->_p->label_layouts);
        }
    }
    $_->forward($is_train) for @{ $self->_p->execs };
}

# Get the shapes of the outputs

method get_output_shapes()
{
    my @shapes = map { $_->shape } @{ $self->execs->[0]->outputs };
    my @concat_shapes;
    zip(sub {
        my ($key, $shape, $axis) = @_;
        my @the_shape = @{ $shape };
        if($axis >= 0)
        {
            $the_shape[$axis] = $self->_p->batch_size;
        }
        push @concat_shapes, AI::MXNet::DataDesc->new(name => $key, shape => \@the_shape);
    }, $self->symbol->list_outputs, \@shapes, $self->_p->output_layouts);
    return \@concat_shapes;
}

=head2 get_outputs

    Gets outputs of the previous forward computation.

    Parameters
    ----------
    merge_multi_context : bool
    Default is 1. In the case when data-parallelism is used, the outputs
    will be collected from multiple devices. A 1 value indicates that we
    should merge the collected results so that they look like from a single
    executor.

    Returns
    -------
    If merge_multi_context is 1, it is [$out1, $out2]. Otherwise, it
    is [[$out1_dev1, $out1_dev2], [$out2_dev1, $out2_dev2]]. All the output
    elements are `AI::MXNet::NDArray`.
=cut

method get_outputs(Bool $merge_multi_context=1)
{
    my $outputs;
    for my $i (0..@{ $self->_p->execs->[0]->outputs }-1)
    {
        my @tmp;
        for my $exec (@{ $self->_p->execs })
        {
            push @tmp, $exec->outputs->[$i];
        }
        push @$outputs, \@tmp;
    }
    if($merge_multi_context)
    {
        $outputs = AI::MXNet::Executor::Group::_merge_multi_context($outputs, $self->_p->output_layouts);
    }
    return $outputs;
}

=head2  get_input_grads

    Get the gradients with respect to the inputs of the module.

    Parameters
    ----------
    merge_multi_context : bool
    Default is 1. In the case when data-parallelism is used, the outputs
    will be collected from multiple devices. A 1 value indicates that we
    should merge the collected results so that they look like from a single
    executor.

    Returns
    -------
    If merge_multi_context is 1, it is [$grad1, $grad2]. Otherwise, it
    is [[$grad1_dev1, $grad1_dev2], [$grad2_dev1, $grad2_dev2]]. All the output
    elements are AI::MXNet::NDArray.
=cut

method get_input_grads(Bool $merge_multi_context=1)
{
    confess("assert \$self->inputs_need_grad") unless $self->inputs_need_grad;
    if($merge_multi_context)
    {
        return AI::MXNet::Executor::Group::_merge_multi_context($self->_p->input_grad_arrays, $self->_p->data_layouts);
    }
    return $self->_p->input_grad_arrays;
}

=head2 backward

    Run backward on all devices. A backward should be called after
    a call to the forward function. Backward cannot be called unless
    $self->for_training is 1.

    Parameters
    ----------
    out_grads : NDArray or array ref of NDArray, optional
    Gradient on the outputs to be propagated back.
    This parameter is only needed when bind is called
    on outputs that are not a loss function.
=cut

method backward(Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]] $out_grads=)
{
    confess('re-bind with for_training=1 to run backward') unless $self->for_training;
    $out_grads //= [];
    zip(sub {
        my ($i, $exec, $islice) = @_;
        my @out_grads_slice;
        zip(sub{
            my ($grad, $axis) = @_;
            if($axis >= 0)
            {
                my $og_my_slice = $grad->slice_axis({
                    axis  => $axis,
                    begin => $islice->[0],
                    end   => $islice->[1]
                });
                push @out_grads_slice, $og_my_slice->as_in_context($self->contexts->[$i]);
            }
            else
            {
                push @out_grads_slice, $grad->copyto($self->contexts->[$i]);
            }
        }, $out_grads, $self->_p->output_layouts);
        $exec->backward(\@out_grads_slice);
    }, [0..@{ $self->_p->execs }-1], $self->_p->execs, $self->_p->slices);
}

=head2 update_metric

    Accumulate the performance according to eval_metric on all devices.

    Parameters
    ----------
    eval_metric : AI::MXNet::EvalMetric
        The metric used for evaluation.
    labels : array ref of NDArray
        Typically comes from label of AI::MXNet::DataBatch.
=cut

method update_metric(AI::MXNet::EvalMetric $eval_metric, ArrayRef[AI::MXNet::NDArray] $labels)
{
    zip(sub {
        my ($texec, $islice) = @_;
        my @labels_slice;
        zip(sub {
            my ($label, $axis) = @_;
            if($axis == 0)
            {
                # slicing NDArray along axis 0 can avoid copying
                push @labels_slice, $label->slice([$islice->[0], $islice->[1]-1]);
            }
            elsif($axis > 0)
            {
                my $label_my_slice = $label->slice_axis({
                    axis  => $axis,
                    begin => $islice->[0],
                    end   => $islice->[1]
                })->as_in_context($label->context);
                push @labels_slice, $label_my_slice;
            }
            else
            {
                push @labels_slice, $label;
            }
        }, $labels, $self->_p->label_layouts);
        $eval_metric->update(\@labels_slice, $texec->outputs);
    }, $self->_p->execs, $self->_p->slices);
}

# Internal utility function to bind the i-th executor.

method _bind_ith_exec(
    Int                                         $i,
    ArrayRef[AI::MXNet::DataDesc]               $data_shapes,
    Maybe[ArrayRef[AI::MXNet::DataDesc]]        $label_shapes,
    Maybe[AI::MXNet::DataParallelExecutorGroup] $shared_group
)
{
    my $shared_exec = $shared_group ? $shared_group->_p->execs->[$i] : undef;
    my $context = $self->contexts->[$i];
    my $shared_data_arrays = $self->_p->shared_data_arrays->[$i];
    my %input_shapes = map { $_->name => $_->shape } @{ $data_shapes };
    if(defined $label_shapes)
    {
        %input_shapes = (%input_shapes, map { $_->name => $_->shape } @{ $label_shapes });
    }
    my ($arg_shapes, undef, $aux_shapes) = $self->symbol->infer_shape(%input_shapes);
    confess("shape inference failed") unless defined $arg_shapes;

    my %input_types = map { $_->name => $_->dtype } @{ $data_shapes };
    my ($arg_types, undef, $aux_types) = $self->symbol->infer_type(%input_types);
    confess("type inference failed") unless defined $arg_types;
    my $arg_arrays = [];
    my $grad_arrays = $self->for_training ? {} : undef;

    #Internal helper to get a memory block or re-use by re-shaping
    my $_get_or_reshape = sub {
            my ($name, $shared_data_arrays, $arg_shape, $arg_type, $context, $logger) = @_;
            my $arg_arr;
            if(exists $shared_data_arrays->{$name})
            {
                $arg_arr = $shared_data_arrays->{$name};
                if(product(@{ $arg_arr->shape }) >= product(@{ $arg_shape }))
                {
                    # nice, we can directly re-use this data blob
                    confess("dtypes do not match") 
                        unless $arg_arr->dtype eq $arg_type;
                    $arg_arr = $arg_arr->reshape($arg_shape);
                }
                else
                {
                    $logger->warning(
                        'bucketing: data "%s" has a shape (%s)'
                        .', which is larger than already allocated '
                        .'shape (%s)'
                        .'. Need to re-allocate. Consider putting '
                        .'default_bucket_key to'
                        .' be the bucket taking the largest input for better '
                        .'memory sharing.',
                        $name, join(',', $arg_shape), join(',', $arg_arr->shape)
                    );
                    $arg_arr = AI::MXNet::NDArray->zeros(
                        $arg_shape,
                        ctx => $context,
                        dtype => $arg_type
                    );
                    # replace existing shared array because the new one is bigger
                    $shared_data_arrays->{ $name } = $arg_arr;
                }
            }
            else
            {
                $arg_arr = AI::MXNet::NDArray->zeros(
                    $arg_shape,
                    ctx => $context,
                    dtype => $arg_type
                );
                $shared_data_arrays->{ $name } = $arg_arr;
            }
            return $arg_arr;
    };
    my %param_names = map { $_ => 1 } @{ $self->param_names };
    # create or borrow arguments and gradients
    for my $j (0..@{ $self->_p->arg_names }-1)
    {
        my $name = $self->_p->arg_names->[$j];
        my $arg_arr;
        if(exists $param_names{ $name }) # model parameter
        {
            if(not defined $shared_exec)
            {
                $arg_arr = AI::MXNet::NDArray->zeros(
                    $arg_shapes->[$j],
                    ctx   => $context,
                    dtype => $arg_types->[$j]
                );

                if($self->grad_req->{$name} ne 'null')
                {
                    my $grad_arr = AI::MXNet::NDArray->zeros(
                        $arg_shapes->[$j],
                        ctx   => $context,
                        dtype => $arg_types->[$j]
                    );
                    $grad_arrays->{ $name } = $grad_arr;
                }
            }
            else
            {
                $arg_arr = $shared_exec->arg_dict->{ $name };
                my $arg_arr_shape = $arg_arr->shape;
                my $arg_shape = $arg_shapes->[$j];
                confess "shapes do not match (@$arg_arr_shape) != (@$arg_shape)"
                    unless "@$arg_arr_shape" eq "@$arg_shape";
                my $arg_arr_type = $arg_arr->dtype;
                my $arg_type = $arg_types->[$j];
                confess "types do not match $arg_arr_type) != $arg_type"
                    unless $arg_arr_type eq $arg_type;
                if($self->grad_req->{ $name } ne 'null')
                {
                    $grad_arrays->{ $name } = $shared_exec->grad_dict->{ $name };
                }
            }
        }
        else # data or label
        {
            $arg_arr = $_get_or_reshape->(
                $name, $shared_data_arrays, $arg_shapes->[$j],
                $arg_types->[$j], $context, $self->logger
            );
            if($self->grad_req->{ $name } ne 'null')
            {
                $grad_arrays->{ $name } = $_get_or_reshape->(
                    "grad of $name", $shared_data_arrays,
                    $arg_shapes->[$j], $arg_types->[$j],
                    $context, $self->logger
                );
            }
        }
        # data might also need grad if inputs_need_grad is True
        push @{ $arg_arrays }, $arg_arr;
    }
    # create or borrow aux variables
    my $aux_arrays = [];
    if(not defined $shared_exec)
    {
        zip(sub{
            my ($s, $t) = @_;
            push @{ $aux_arrays }, AI::MXNet::NDArray->zeros($s, ctx => $context, dtype => $t);
        }, $aux_shapes, $aux_types);
    }
    else
    {
        for my $j (0..@{ $shared_exec->aux_arrays }-1)
        {
            my $arr = $shared_exec->aux_arrays->[$j];
            my $aux_shape = $aux_shapes->[$j];
            my $arr_shape = $arr->shape;
            confess("aux shape (@$aux_shape) != array shape (@$arr_shape)")
                unless "@$aux_shape" eq "@$arr_shape";
            my $aux_type = $aux_types->[$j];
            my $arr_type = $arr->dtype;
            confess("aux_type $aux_type != array type $arr_type")
                unless $aux_type ne $arr_type;
        }
        @{ $aux_arrays } = @{ $shared_exec->aux_arrays };
    }
    my $executor = $self->symbol->bind(
        ctx => $context, args => $arg_arrays,
        args_grad => $grad_arrays, aux_states => $aux_arrays,
        grad_req => $self->grad_req, shared_exec => $shared_exec
    );
    return $executor;
}

=head2 _sliced_shape

    Get the sliced shapes for the i-th executor.

    Parameters
    ----------
    shapes : array ref of (str, array ref)
        The original (name, shape) pairs.
    i : int
    Which executor we are dealing with.
=cut

method _sliced_shape(ArrayRef[AI::MXNet::DataDesc] $shapes, Int $i, ArrayRef[Int] $major_axis)
{
    my @sliced_shapes;
    zip(sub {
        my ($desc, $axis) = @_;
        my @shape = @{ $desc->shape };
        if($axis >= 0)
        {
            $shape[$axis] = $self->_p->slices->[$i]->[1] - $self->_p->slices->[$i]->[0];
        }
        push @sliced_shapes, AI::MXNet::DataDesc->new(
            name    => $desc->name,
            shape   => \@shape,
            dtype   => $desc->dtype,
            layout  => $desc->layout
        );
    }, $shapes, $major_axis);
    return \@sliced_shapes;
}

=head2 install_monitor

    Install monitor on all executors

    Parameters
    ----------
    $mon : AI::MXNet::Monitor
=cut

method install_monitor(AI::MXNet::Monitor $mon)
{
    $mon->install($_) for @{ $self->_p->execs };
}

1;
