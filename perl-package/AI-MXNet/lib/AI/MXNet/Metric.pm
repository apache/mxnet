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

package AI::MXNet::Metric;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use Scalar::Util qw/blessed/;
use JSON::PP;

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
has '_kwargs'    => (is => 'rw', init_arg => undef);
around BUILDARGS => \&AI::MXNet::Base::process_arguments;

sub BUILD
{
    my ($self, $kwargs) = @_;
    $self->reset;
    $self->_kwargs($kwargs);
}

method _class_name()
{
    my $class = ref $self || $self;
    $class =~ s/^.+:://;
    $class;
}

=head2 get_config

    Save configurations of metric. Can be recreated
        from configs with mx->metric->create(%{ $config })
=cut

method get_config()
{
    my %config = %{ $self->_kwargs };
    %config = (%config,
        metric => $self->_class_name,
        name   => $self->name
    );
    return \%config;
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
method python_constructor_arguments() { ['metrics'] }

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
    for(zip($labels, $preds)) {
        my ($label, $pred_label) = @$_;
        if(join(',', @{$pred_label->shape}) ne join(',', @{$label->shape}))
        {
            $pred_label = AI::MXNet::NDArray->argmax_channel($pred_label);
        }
        AI::MXNet::Metric::check_label_shapes($label, $pred_label);
        my $sum = ($pred_label->aspdl->flat == $label->aspdl->flat)->sum;
        $self->sum_metric($self->sum_metric + $sum);
        $self->num_inst($self->num_inst + $pred_label->size);
    }
}

package AI::MXNet::TopKAccuracy;
use Mouse;
use List::Util qw/min/;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'top_k_accuracy');
has 'top_k' => (is => 'rw', isa => 'int', default => 1);
method python_constructor_arguments() { ['top_k'] }

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
    for(zip($labels, $preds)) {
        my ($label, $pred_label) = @$_;
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
    }
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
    for(zip($labels, $preds)) {
        my ($label, $pred_label) = @$_;
        AI::MXNet::Metric::check_label_shapes($label, $pred_label);
        $pred_label = $pred_label->aspdl->maximum_ind;
        $label = $label->astype('int32')->aspdl;
        confess("F1 currently only supports binary classification.")
            if $label->uniq->shape->at(0) > 2;
        my ($true_positives, $false_positives, $false_negatives) = (0,0,0);
        for(zip($pred_label->unpdl, $label->unpdl)) {
            my ($y_pred, $y_true) = @$_;
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
        }
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
    }
}

package AI::MXNet::Perplexity;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'        => (default => 'Perplexity');
has 'ignore_label' => (is => 'ro', isa => 'Maybe[Int]');
has 'axis'         => (is => 'ro', isa => 'Int', default => -1);
method python_constructor_arguments() { ['ignore_label', 'axis'] }

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
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
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
    }
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
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + ($label - $pred)->abs->avg);
        $self->num_inst($self->num_inst + 1);
    }
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
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + (($label - $pred)**2)->avg);
        $self->num_inst($self->num_inst + 1);
    }
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
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        if($label->ndims == 1)
        {
            $label = $label->reshape(1, $label->shape->at(0));
        }
        $self->sum_metric($self->sum_metric + sqrt((($label - $pred)**2)->avg));
        $self->num_inst($self->num_inst + 1);
    }
}

# Calculate Cross Entropy loss
package AI::MXNet::CrossEntropy;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'cross-entropy');
has 'eps'     => (is => 'ro', isa => 'Num', default => 1e-12);
method python_constructor_arguments() { ['eps'] }

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
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
    }
}

package AI::MXNet::PearsonCorrelation;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'pearson-correlation');

=head1 NAME

    AI::MXNet::PearsonCorrelation
=cut

=head1 DESCRIPTION

    Computes Pearson correlation.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.

    Examples
    --------
    >>> $predicts = [mx->nd->array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> $labels   = [mx->nd->array([[1, 0], [0, 1], [0, 1]])]
    >>> $pr = mx->metric->PearsonCorrelation()
    >>> $pr->update($labels, $predicts)
    >>> print pr->get()
    ('pearson-correlation', '0.421637061887229')
=cut

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        AI::MXNet::Metric::check_label_shapes($label, $pred);
        $label = $label->aspdl->flat;
        $pred  = $pred->aspdl->flat;
        my ($label_mean, $label_stdv) = ($label->stats)[0, 6];
        my ($pred_mean, $pred_stdv) = ($pred->stats)[0, 6];
        $self->sum_metric(
            $self->sum_metric
                +
            ((($label-$label_mean)*($pred-$pred_mean))->sum/$label->nelem)/(($label_stdv*$pred_stdv)->at(0))
        );
        $self->num_inst($self->num_inst + 1);
    }
}

package AI::MXNet::Loss;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'loss');

=head1 NAME

    AI::MXNet::Loss
=cut

=head1 DESCRIPTION

    Dummy metric for directly printing loss.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
=cut

method update($labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    for my $pred (@{ $preds })
    {
        $self->sum_metric($self->sum_metric + $pred->sum->asscalar);
        $self->num_inst($self->num_inst + $pred->size);
    }
}

package AI::MXNet::Confidence;
use Mouse;

=head1 NAME

    AI::MXNet::Confidence
=cut

=head1 DESCRIPTION

    Accuracy by confidence buckets.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    num_classes: Int
        number of classes
    confidence_thresholds: ArrayRef[Num]
        confidence buckets
    For example
    my $composite_metric  = AI::MXNet::CompositeEvalMetric->new;
    $composite_metric->add(mx->metric->create('acc'));
    $composite_metric->add(
        AI::MXNet::Confidence->new(
            num_classes => 2,
            confidence_thresholds => [ 0.5, 0.7, 0.8, 0.9 ],
        )
    );
=cut

extends 'AI::MXNet::EvalMetric';
has 'num_classes', is => 'ro', isa => 'Int', required => 1;
has 'confidence_thresholds', is => 'ro', isa => 'ArrayRef[Num]', required => 1;
has '+name'   => (default => 'confidence');
has '+sum_metric', isa => 'PDL';
has '+num_inst', isa => 'PDL';
method python_constructor_arguments() { ['num_classes', 'confidence_thresholds'] }

sub _hot
{
    my($m, $n) = @_;
    my $md = $m->dim(-1);
    my $hot = PDL->zeros($n, $md);
    $hot->index2d($m->flat(), PDL->sequence($md)) .= 1;
    return $hot;
}

sub reset
{
    my($self) = @_;
    my $nt = @{$self->confidence_thresholds};
    my $n = $self->num_classes;
    $self->sum_metric(PDL->zeroes($nt, $n));
    $self->num_inst(PDL->zeroes($nt, $n));
    return;
}

sub update
{
    my($self, $labels, $preds) = @_;
    my $n = $self->num_classes;
    my $ct = PDL->new($self->confidence_thresholds);
    my $nt = $ct->nelem;
    for(0 .. @$labels - 1)
    {
        my $label = _hot($labels->[$_]->aspdl, $n);
        my $pred = $preds->[$_]->aspdl;
        for my $c (0 .. $n - 1)
        {
            my $ls = $label->slice($c);
            my $pm = $pred->slice($c) > $ct;
            $self->sum_metric->slice(":,$c") += ($pm & $ls);
            $self->num_inst->slice(":,$c") += $pm;
        }
    }
    return;
}

sub get
{
    my($self) = @_;
    my(@names, @values);
    my $val = $self->sum_metric / $self->num_inst;
    my $ct = $self->confidence_thresholds;
    my $n = $self->num_classes;
    for my $c (0 .. $n - 1)
    {
        for my $t (0 .. @$ct - 1)
        {
            my $sm = $self->sum_metric->at($t, $c);
            my $ni = $self->num_inst->at($t, $c);
            push @names, "P(v=$c|Conf>$ct->[$t])=($sm/$ni)";
            push @values, $val->at($t, $c);
        }
    }
    return(\@names, \@values);
}

=head1 NAME

    AI::MXNet::CustomMetric
=cut

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
method python_constructor_arguments() { ['eval_function', 'allow_extra_outputs'] }

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds)
        unless $self->allow_extra_outputs;
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        $label = $label->aspdl;
        $pred =  $pred->aspdl;
        my $value = $self->eval_function->($label, $pred);
        my $sum_metric = ref $value ? $value->[0] : $value;
        my $num_inst   = ref $value ? $value->[1] : 1;
        $self->sum_metric($self->sum_metric + $sum_metric);
        $self->num_inst($self->num_inst + $num_inst);
    }
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
    acc                 AI::MXNet::Accuracy
    accuracy            AI::MXNet::Accuracy
    ce                  AI::MXNet::CrossEntropy
    crossentropy        AI::MXNet::CrossEntropy
    f1                  AI::MXNet::F1
    mae                 AI::MXNet::MAE
    mse                 AI::MXNet::MSE
    rmse                AI::MXNet::RMSE
    top_k_accuracy      AI::MXNet::TopKAccuracy
    topkaccuracy        AI::MXNet::TopKAccuracy
    perplexity          AI::MXNet::Perplexity
    pearsonr            AI::MXNet::PearsonCorrelation
    pearsoncorrelation  AI::MXNet::PearsonCorrelation
    loss                AI::MXNet::Loss
    compositeevalmetric AI::MXNet::CompositeEvalMetric
    confidence          AI::MXNet::Confidence
/;

method create(Metric|ArrayRef[Metric] $metric, @kwargs)
{
    Carp::confess("metric must be defined") unless defined $metric;
    return $metric if blessed $metric and $metric->isa('AI::MXNet::EvalMetric');
    if(my $ref = ref $metric)
    {
        if($ref eq 'ARRAY')
        {
            my $composite_metric = AI::MXNet::CompositeEvalMetric->new();
            for my $child_metric (@{ $metric })
            {
                $composite_metric->add(__PACKAGE__->create($child_metric, @kwargs))
            }
            return $composite_metric;
        }
        else
        {
            return AI::MXNet::CustomMetric->new(eval_function => $metric, @kwargs);
        }
    }
    else
    {
        if(not exists $metrics{ lc($metric) } and not $metric =~ /^{/)
        {
            my @metrics = keys %metrics;
            Carp::confess("Metric must be either subref or one of [@metrics]");
        }
        if($metric =~ /^{/ and not @kwargs)
        {
            my $config = decode_json($metric);
            $metric = delete $config->{metric};
            @kwargs = %{ $config };
        }
        return $metrics{ lc($metric) }->new(@kwargs);
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
