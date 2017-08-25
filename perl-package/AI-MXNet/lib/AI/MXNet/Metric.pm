package AI::MXNet::Metric;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use Scalar::Util qw/blessed/;

=head1 NAME

    AI::MXNet::Metric - Online evaluation metric module.
=cut

# Check to see if the two arrays are the same size.
sub _calculate_shape
{
    my $input = shift;
    my ($shape);
    if(blessed($input))
    {
        if($input->isa('PDL'))
        {
            $shape = $input->shape->at(-1);
        }
        else
        {
            $shape = $input->shape->[0];
        }
    }
    else
    {
        $shape = @{ $input };
    }
    return $shape;
}
func check_label_shapes(
    ArrayRef|AI::MXNet::NDArray|PDL $labels,
    ArrayRef|AI::MXNet::NDArray|PDL $preds
)
{
    my ($label_shape, $pred_shape) = (_calculate_shape($labels), _calculate_shape($preds));
    Carp::confess(
        "Shape of labels $label_shape does not "
        ."match shape of predictions $pred_shape"
    ) unless $pred_shape == $label_shape;
}

=head1 DESCRIPTION

    Base class of all evaluation metrics.
=cut

package AI::MXNet::EvalMetric;
use Mouse;
use overload '""' => sub {
    return "EvalMetric: "
            .Data::Dumper->new(
                [shift->get_name_value()]
            )->Purity(1)->Deepcopy(1)->Terse(1)->Dump
},  fallback => 1;
has 'name'       => (is => 'rw', isa => 'Str');
has 'num'        => (is => 'rw', isa => 'Int');
has 'num_inst'   => (is => 'rw', isa => 'Maybe[Int|ArrayRef[Int]]');
has 'sum_metric' => (is => 'rw', isa => 'Maybe[Num|ArrayRef[Num]]');

sub BUILD
{
    shift->reset;
}

method update($label, $pred)
{
    confess('NotImplemented');
}

method reset()
{
    if(not defined $self->num)
    {
        $self->num_inst(0);
        $self->sum_metric(0);
    }
    else
    {
        $self->num_inst([(0) x $self->num]);
        $self->sum_metric([(0) x $self->num]);
    }
}

method get()
{
    if(not defined $self->num)
    {
        if($self->num_inst == 0)
        {
            return ($self->name, 'nan');
        }
        else
        {
            return ($self->name, $self->sum_metric / $self->num_inst);
        }
    }
    else
    {
        my $names = [map { sprintf('%s_%d', $self->name, $_) } 0..$self->num-1];
        my $values = [];
        for (my $i = 0; $i < @{ $self->sum_metric }; $i++)
        {
            my ($x, $y) = ($self->sum_metric->[$i], $self->num_inst->[$i]);
            if($y != 0)
            {
                push (@$values, $x/$y);
            }
            else
            {
                push (@$values, 'nan');
            }
        }
        return ($names, $values);
    }
}

method get_name_value()
{
    my ($name, $value) = $self->get;
    $name = [$name] unless ref $name;
    $value = [$value] unless ref $value;
    my %ret;
    @ret{ @$name } = @$value;
    return \%ret;
}

package AI::MXNet::CompositeEvalMetric;
use Mouse;

extends 'AI::MXNet::EvalMetric';
has 'metrics' => (is => 'rw', isa => 'ArrayRef[AI::MXNet::EvalMetric]', default => sub { [] });
has '+name'   => (default => 'composite');

# Add a child metric.
method add(AI::MXNet::EvalMetric $metric)
{
    push @{ $self->metrics }, $metric;
}

# Get a child metric.
method get_metric(int $index)
{
    my $max = @{ $self->metrics } - 1;
    confess("Metric index $index is out of range 0 and $max")
        if $index > $max;
    return $self->metrics->[$index];
}

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    for my $metric (@{ $self->metrics })
    {
        $metric->update($labels, $preds);
    }
}

method reset()
{
    for my $metric (@{ $self->metrics })
    {
        $metric->reset;
    }
}

method get()
{
    my $names = [];
    my $results = [];
    for my $metric (@{ $self->metrics })
    {
        my ($name, $result) = $metric->get;
        $name = [$name] unless ref $name;
        $result = [$result] unless ref $result;
        push @$names, @$name;
        push @$results, @$result;
    }
    return ($names, $results);
}


########################
# CLASSIFICATION METRICS
########################

package AI::MXNet::Accuracy;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'accuracy');

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred_label) = @_;
        if(join(',', @{$pred_label->shape}) ne join(',', @{$label->shape}))
        {
            $pred_label = AI::MXNet::NDArray->argmax_channel($pred_label);
        }
        AI::MXNet::Metric::check_label_shapes($label, $pred_label);
        my $sum = ($pred_label->aspdl->flat == $label->aspdl->flat)->sum;
        $self->sum_metric($self->sum_metric + $sum);
        $self->num_inst($self->num_inst + $pred_label->size);
    }, $labels, $preds);
}

package AI::MXNet::TopKAccuracy;
use Mouse;
use List::Util qw/min/;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'top_k_accuracy');
has 'top_k' => (is => 'rw', isa => 'int', default => 1);

sub BUILD
{
    my $self = shift;
    confess("Please use Accuracy if top_k is no more than 1")
        unless $self->top_k > 1;
    $self->name($self->name . "_" . $self->top_k);
}

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred_label) = @_;
        confess('Predictions should be no more than 2 dims')
            unless @{ $pred_label->shape } <= 2;
        $pred_label = $pred_label->aspdl->qsorti;
        $label = $label->astype('int32')->aspdl;
        AI::MXNet::Metric::check_label_shapes($label, $pred_label);
        my $num_samples = $pred_label->shape->at(-1);
        my $num_dims = $pred_label->ndims;
        if($num_dims == 1)
        {
            my $sum = ($pred_label->flat == $label->flat)->sum;
            $self->sum_metric($self->sum_metric + $sum);
        }
        elsif($num_dims == 2)
        {
            my $num_classes = $pred_label->shape->at(0);
            my $top_k = min($num_classes, $self->top_k);
            for my $j (0..$top_k-1)
            {
                my $sum = ($pred_label->slice($num_classes -1 - $j, 'X')->flat == $label->flat)->sum;
                $self->sum_metric($self->sum_metric + $sum);
            }
        }
        $self->num_inst($self->num_inst + $num_samples);
    }, $labels, $preds);
}

# Calculate the F1 score of a binary classification problem.
package AI::MXNet::F1;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'f1');

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred_label) = @_;
        AI::MXNet::Metric::check_label_shapes($label, $pred_label);
        $pred_label = $pred_label->aspdl->maximum_ind;
        $label = $label->astype('int32')->aspdl;
        confess("F1 currently only supports binary classification.")
            if $label->uniq->shape->at(0) > 2;
        my ($true_positives, $false_positives, $false_negatives) = (0,0,0);
        zip(sub{
            my ($y_pred, $y_true) = @_;
            if($y_pred == 1 and $y_true == 1)
            {
                $true_positives += 1;
            }
            elsif($y_pred == 1 and $y_true == 0)
            {
                $false_positives += 1;
            }
            elsif($y_pred == 0 and $y_true == 1)
            {
                $false_negatives += 1;
            }
        }, $pred_label->unpdl, $label->unpdl);
        my $precision;
        my $recall;
        if($true_positives + $false_positives > 0)
        {
            $precision = $true_positives / ($true_positives + $false_positives);
        }
        else
        {
            $precision = 0;
        }
        if($true_positives + $false_negatives > 0)
        {
            $recall = $true_positives / ($true_positives +  $false_negatives);
        }
        else
        {
            $recall = 0;
        }
        my $f1_score;
        if($precision + $recall > 0)
        {
            $f1_score = 2 * $precision * $recall / ($precision + $recall);
        }
        else
        {
            $f1_score = 0;
        }
        $self->sum_metric($self->sum_metric + $f1_score);
        $self->num_inst($self->num_inst + 1);
    }, $labels, $preds);
}

package AI::MXNet::Perplexity;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'        => (default => 'Perplexity');
has 'ignore_label' => (is => 'ro', isa => 'Maybe[Int]');
has 'axis'         => (is => 'ro', isa => 'Int', default => -1);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(ignore_label => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

=head1 NAME

    AI::MXNet::Perplexity
=cut

=head1 DESCRIPTION

    Calculate perplexity.

    Parameters
    ----------
    ignore_label : int or undef
        index of invalid label to ignore when
        counting. usually should be -1. Include
        all entries if undef.
    axis : int (default -1)
        The axis from prediction that was used to
        compute softmax. By default uses the last
        axis.
=cut

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    my ($loss, $num) = (0, 0);
    zip(sub {
        my ($label, $pred) = @_;
        my $label_shape = $label->shape;
        my $pred_shape  = $pred->shape;
        assert(
            (product(@{ $label_shape }) == product(@{ $pred_shape })/$pred_shape->[-1]),
            "shape mismatch: (@$label_shape) vs. (@$pred_shape)"
        );
        $label = $label->as_in_context($pred->context)->reshape([$label->size]);
        $pred = AI::MXNet::NDArray->pick($pred, $label->astype('int32'), { axis => $self->axis });
        if(defined $self->ignore_label)
        {
            my $ignore = ($label == $self->ignore_label);
            $num -= $ignore->sum->asscalar;
            $pred = $pred*(1-$ignore) + $ignore;
        }
        $loss -= $pred->maximum(1e-10)->log->sum->asscalar;
        $num  += $pred->size;
    }, $labels, $preds);
    $self->sum_metric($self->sum_metric + $loss);
    $self->num_inst($self->num_inst + $num);
}

method get()
{
    return ($self->name, exp($self->sum_metric / $self->num_inst));
}

####################
# REGRESSION METRICS
####################

# Calculate Mean Absolute Error loss
package AI::MXNet::MAE;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'mae');

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred) = @_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + ($label - $pred)->abs->avg);
        $self->num_inst($self->num_inst + 1);
    }, $labels, $preds);
}

# Calculate Mean Squared Error loss
package AI::MXNet::MSE;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'mse');

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred) = @_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + (($label - $pred)**2)->avg);
        $self->num_inst($self->num_inst + 1);
    }, $labels, $preds);
}

# Calculate Root Mean Squred Error loss
package AI::MXNet::RMSE;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'rmse');

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred) = @_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + sqrt((($label - $pred)**2)->avg));
        $self->num_inst($self->num_inst + 1);
    }, $labels, $preds);
}

# Calculate Cross Entropy loss
package AI::MXNet::CrossEntropy;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'cross-entropy');
has 'eps'     => (is => 'ro', isa => 'Num', default => 1e-8);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(eps => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    zip(sub {
        my ($label, $pred) = @_;
        $label = $label->aspdl->flat;
        $pred =  $pred->aspdl;
        my $label_shape = $label->shape->at(0);
        my $pred_shape  = $pred->shape->at(-1);
        confess(
            "Size of label  $label_shape and 
            .first dimension of pred $pred_shape do not match"
        ) unless $label_shape == $pred_shape;
        my $prob = $pred->index($label);
        $self->sum_metric($self->sum_metric + (-($prob + $self->eps)->log)->sum);
        $self->num_inst($self->num_inst + $label_shape);
    }, $labels, $preds);
}

=head1 DESCRIPTION

    Custom evaluation metric that takes a sub ref.

    Parameters
    ----------
    eval_function : subref
        Customized evaluation function.
    name : str, optional
        The name of the metric
    allow_extra_outputs : bool
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding.
=cut


package AI::MXNet::CustomMetric;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has 'eval_function'       => (is => 'ro', isa => 'CodeRef');
has 'allow_extra_outputs' => (is => 'ro', isa => 'Int', default => 0);

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds)
        unless $self->allow_extra_outputs;
    zip(sub {
        my ($label, $pred) = @_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        my $value = $self->eval_function->($label, $pred);
        my $sum_metric = ref $value ? $value->[0] : $value;
        my $num_inst   = ref $value ? $value->[1] : 1;
        $self->sum_metric($self->sum_metric + $sum_metric);
        $self->num_inst($self->num_inst + $num_inst);
    }, $labels, $preds);
}

package AI::MXNet::Metric;

=head2 create

    Create an evaluation metric.

    Parameters
    ----------
    metric : str or sub ref
        The name of the metric, or a function
        providing statistics given pred, label NDArray.
=cut

my %metrics = qw/
    acc            AI::MXNet::Accuracy
    accuracy       AI::MXNet::Accuracy
    ce             AI::MXNet::CrossEntropy
    f1             AI::MXNet::F1
    mae            AI::MXNet::MAE
    mse            AI::MXNet::MSE
    rmse           AI::MXNet::RMSE
    top_k_accuracy AI::MXNet::TopKAccuracy
    Perplexity     AI::MXNet::Perplexity
    perplexity     AI::MXNet::Perplexity
/;

method create(Metric|ArrayRef[Metric] $metric, %kwargs)
{
    Carp::confess("metric must be defined") unless defined $metric;
    if(my $ref = ref $metric)
    {
        if($ref eq 'ARRAY')
        {
            my $composite_metric = AI::MXNet::CompositeEvalMetric->new();
            for my $child_metric (@{ $metric })
            {
                $composite_metric->add(__PACKAGE__->create($child_metric, %kwargs))
            }
            return $composite_metric;
        }
        else
        {
            return AI::MXNet::CustomMetric->new(eval_function => $metric, %kwargs);
        }
    }
    else
    {
        if(not exists $metrics{ lc($metric) })
        {
            my @metrics = keys %metrics;
            Carp::confess("Metric must be either subref or one of [@metrics]");
        }
        return $metrics{ lc($metric) }->new(%kwargs);
    }
}

{
    no strict 'refs';
    no warnings 'redefine';
    for my $metric (values %metrics)
    {
        my ($name) = $metric =~ /(\w+)$/;
        *{__PACKAGE__."::$name"} = sub { shift; $metric->new(@_); };
    }
}

1;