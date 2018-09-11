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

    AI::MXNet::Metric - Evaluation Metric API.
=head1 DESCRIPTION

    This module hosts all the evaluation metrics available to evaluate the performance of a learned model.
    L<Python Docs|http://mxnet.incubator.apache.org/api/python/metric/metric.html>
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

=head1 NAME

    AI::MXNet::Accuracy - Computes accuracy classification score.
=cut

=head1 DESCRIPTION

    The accuracy score is defined as

    accuracy(y, y^) = (1/n) * sum(i=0..n−1) { y^(i)==y(i) }

    Parameters:
    axis (Int, default=1) – The axis that represents classes.
    name (Str, default='accuracy') – Name of this metric instance for display.

    pdl> use AI::MXNet qw(mx)
    pdl> $predicts = [mx->nd->array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    pdl> $labels   = [mx->nd->array([[0, 1, 1]])]
    pdl> $acc = mx->metric->Accuracy()
    pdl> $acc->update($labels, $predicts)
    pdl> use Data::Dumper
    pdl> print Dumper([$acc->get])
    $VAR1 = [
          'accuracy',
          '0.666666666666667'
    ];

=cut

package AI::MXNet::Accuracy;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'accuracy');
has 'axis'    => (is => 'ro', isa => 'Int', default => 1);

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    for(zip($labels, $preds)) {
        my ($label, $pred_label) = @$_;
        if(join(',', @{$pred_label->shape}) ne join(',', @{$label->shape}))
        {
            $pred_label = AI::MXNet::NDArray->argmax_channel($pred_label, { axis => $self->axis });
        }
        my $sum = ($pred_label->aspdl->flat == $label->aspdl->flat)->sum;
        $self->sum_metric($self->sum_metric + $sum);
        $self->num_inst($self->num_inst + $pred_label->size);
    }
}

=head1 NAME

    AI::MXNet::TopKAccuracy - Computes top k predictions accuracy.
=cut

=head1 DESCRIPTION

    TopKAccuracy differs from Accuracy in that it considers the prediction
    to be True as long as the ground truth label is in the top K predicated labels.

    If top_k = 1, then TopKAccuracy is identical to Accuracy.

    Parameters:	
    top_k(Int, default 1) – Whether targets are in top k predictions.
    name (Str, default 'top_k_accuracy') – Name of this metric instance for display.

    use AI::MXNet qw(mx);
    $top_k = 3;
    $predicts = [mx->nd->array(
      [[0.80342804, 0.5275223 , 0.11911147, 0.63968144, 0.09092526,
        0.33222568, 0.42738095, 0.55438581, 0.62812652, 0.69739294],
       [0.78994969, 0.13189035, 0.34277045, 0.20155961, 0.70732423,
        0.03339926, 0.90925004, 0.40516066, 0.76043547, 0.47375838],
       [0.28671892, 0.75129249, 0.09708994, 0.41235779, 0.28163896,
        0.39027778, 0.87110921, 0.08124512, 0.55793117, 0.54753428],
       [0.33220307, 0.97326881, 0.2862761 , 0.5082575 , 0.14795074,
        0.19643398, 0.84082001, 0.0037532 , 0.78262101, 0.83347772],
       [0.93790734, 0.97260166, 0.83282304, 0.06581761, 0.40379256,
        0.37479349, 0.50750135, 0.97787696, 0.81899021, 0.18754124],
       [0.69804812, 0.68261077, 0.99909815, 0.48263116, 0.73059268,
        0.79518236, 0.26139168, 0.16107376, 0.69850315, 0.89950917],
       [0.91515562, 0.31244902, 0.95412616, 0.7242641 , 0.02091039,
        0.72554552, 0.58165923, 0.9545687 , 0.74233195, 0.19750339],
       [0.94900651, 0.85836332, 0.44904621, 0.82365038, 0.99726878,
        0.56413064, 0.5890016 , 0.42402702, 0.89548786, 0.44437266],
       [0.57723744, 0.66019353, 0.30244304, 0.02295771, 0.83766937,
        0.31953292, 0.37552193, 0.18172362, 0.83135182, 0.18487429],
       [0.96968683, 0.69644561, 0.60566253, 0.49600661, 0.70888438,
        0.26044186, 0.65267488, 0.62297362, 0.83609334, 0.3572364 ]]
    )];
    $labels = [mx->nd->array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])];
    $acc = mx->metric->TopKAccuracy(top_k=>$top_k);
    $acc->update($labels, $predicts);
    use Data::Dumper;
    print Dumper([$acc->get]);
    $VAR1 = [
          'top_k_accuracy_3',
          '0.3'
    ];


=cut

package AI::MXNet::TopKAccuracy;
use Mouse;
use List::Util qw/min/;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'top_k_accuracy');
has 'top_k' => (is => 'rw', isa => 'Int', default => 1);
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

package _BinaryClassificationMetrics {
    use Mouse;
    #Private container class for classification metric statistics. True/false positive and
    # true/false negative counts are sufficient statistics for various classification metrics.
    #This class provides the machinery to track those statistics across mini-batches of
    #(label, prediction) pairs.
    has [qw/true_positives
            false_negatives
            false_positives
            true_negatives/] => (is => 'rw', isa => 'Int', default => 0);

    method update_binary_stats(AI::MXNet::NDArray $label, AI::MXNet::NDArray $pred)
    {
        $pred = AI::MXNet::NDArray->argmax($pred, { axis => 1 })->aspdl;
        $label = $label->astype('int32')->aspdl;

        AI::MXNet::Metric::check_label_shapes($label, $pred);
        if($label->uniq->len > 2)
        {
            confess("Currently only support binary classification.");
        }

        my $pred_true = ($pred == 1);
        my $pred_false = 1 - $pred_true;
        my $label_true = ($label == 1);
        my $label_false = 1 - $label_true;

        $self->true_positives($self->true_positives + ($pred_true * $label_true)->sum);
        $self->false_positives($self->false_positives + ($pred_true * $label_false)->sum);
        $self->false_negatives($self->false_negatives + ($pred_false * $label_true)->sum);
        $self->true_negatives($self->true_negatives + ($pred_false * $label_false)->sum);
    }

    method precision()
    {
        if($self->true_positives + $self->false_positives > 0)
        {
            return $self->true_positives / ($self->true_positives + $self->false_positives);
        }
        else
        {
            return 0;
        }
    }

    method recall()
    {
        if($self->true_positives + $self->false_negatives > 0)
        {
            return $self->true_positives / ($self->true_positives + $self->false_negatives);
        }
        else
        {
            return 0;
        }
    }

    method fscore()
    {
        if($self->precision + $self->recall > 0)
        {
            return 2 * $self->precision * $self->recall / ($self->precision + $self->recall);
        }
        else
        {
            return 0;
        }
    }

    method matthewscc()
    {
        if(not $self->total_examples)
        {
            return 0;
        }
        my @terms = (
            $self->true_positives + $self->false_positives,
            $self->true_positives + $self->false_negatives,
            $self->true_negatives + $self->false_positives,
            $self->true_negatives + $self->false_negatives
        );
        my $denom = 1;
        for my $t (grep { $_ } @terms)
        {
            $denom *= $t;
        }
        return (($self->true_positives * $self->true_negatives) - ($self->false_positives * $self->false_negatives)) / sqrt($denom);
    }

    method total_examples()
    {
        return $self->false_negatives + $self->false_positives +
               $self->true_negatives + $self->true_positives;
    }

    method reset_stats()
    {
        $self->false_positives(0);
        $self->false_negatives(0);
        $self->true_positives(0);
        $self->true_negatives(0);
    }
};

=head1 NAME

    AI::MXNet::F1 - Calculate the F1 score of a binary classification problem.
=cut

=head1 DESCRIPTION

    The F1 score is equivalent to harmonic mean of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for F1 score is:

    F1 = 2 * (precision * recall) / (precision + recall)
    The formula for precision and recall is:

    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)
    Note:

    This F1 score only supports binary classification.

    Parameters:
    name (Str, default 'f1') – Name of this metric instance for display.
    average (Str, default 'macro') –
    Strategy to be used for aggregating across mini-batches.
    “macro”: average the F1 scores for each batch. “micro”: compute a single F1 score across all batches.


    $predicts = [mx.nd.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])];
    $labels   = [mx.nd.array([0., 1., 1.])];
    $f1 = mx->metric->F1();
    $f1->update($labels, $predicts);
    print $f1->get;
    f1 0.8

=cut

package AI::MXNet::F1;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'f1');
has 'average' => (is => 'ro', isa => 'Str', default => 'macro');
has 'metrics' => (is => 'rw', init_arg => undef, default => sub { _BinaryClassificationMetrics->new });
has 'method'  => (is => 'ro', init_arg => undef, default => 'fscore');
method python_constructor_arguments() { [qw/name average/] }

method update(ArrayRef[AI::MXNet::NDArray] $labels, ArrayRef[AI::MXNet::NDArray] $preds)
{
    my $method = $self->method;
    AI::MXNet::Metric::check_label_shapes($labels, $preds);
    for(zip($labels, $preds)) {
        my ($label, $pred) = @$_;
        $self->metrics->update_binary_stats($label, $pred);
        if($self->average eq "macro")
        {
            $self->sum_metric($self->sum_metric + $self->metrics->$method);
            $self->num_inst($self->num_inst + 1);
            $self->metrics->reset_stats();
        }
        else
        {
            $self->sum_metric($self->metrics->fscore * $self->metrics->total_examples);
            $self->num_inst($self->metrics->total_examples);
        }
    }
}

method reset()
{
    $self->sum_metric(0);
    $self->num_inst(0);
    $self->metrics->reset_stats();
}

=head1 NAME

    AI::MXNet::MCC - Computes the Matthews Correlation Coefficient of a binary classification problem.
=cut

=head1 DESCRIPTION

    While slower to compute than F1 the MCC can give insight that F1 or Accuracy cannot.
    For instance, if the network always predicts the same result
    then the MCC will immeadiately show this. The MCC is also symetric with respect
    to positive and negative categorization, however, there needs to be both
    positive and negative examples in the labels or it will always return 0.
    MCC of 0 is uncorrelated, 1 is completely correlated, and -1 is negatively correlated.

        MCC = (TP * TN - FP * FN)/sqrt( (TP + FP)*( TP + FN )*( TN + FP )*( TN + FN ) )

    where 0 terms in the denominator are replaced by 1.

    This version of MCC only supports binary classification.

    Parameters
    ----------
    name : str, 'mcc'
        Name of this metric instance for display.
    average : str, default 'macro'
        Strategy to be used for aggregating across mini-batches.
            "macro": average the MCC for each batch.
            "micro": compute a single MCC across all batches.

    Examples
    --------
    In this example the network almost always predicts positive
    >>> $false_positives = 1000
    >>> $false_negatives = 1
    >>> $true_positives = 10000
    >>> $true_negatives = 1
    >>> $predicts = [mx->nd->array(
        [
            ([.3, .7])x$false_positives,
            ([.7, .3])x$true_negatives,
            ([.7, .3])x$false_negatives,
            ([.3, .7])xtrue_positives
        ]
    )];
    >>> $labels  = [mx->nd->array(
        [
            (0)x($false_positives + $true_negatives),
            (1)x($false_negatives + $true_positives)
        ]
    )];
    >>> $f1 = mx->metric->F1();
    >>> $f1->update($labels, $predicts);
    >>> $mcc = mx->metric->MCC()
    >>> $mcc->update($labels, $predicts)
    >>> print $f1->get();
    f1 0.95233560306652054
    >>> print $mcc->get();
    mcc 0.01917751877733392

=cut

package AI::MXNet::MCC;
use Mouse;
extends 'AI::MXNet::F1';
has '+name'   => (default => 'mcc');
has '+method' => (default => 'matthewscc');

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

    AI::MXNet::Perplexity - Calculate perplexity.
=cut

=head1 DESCRIPTION

    Perplexity is a measurement of how well a probability distribution or model predicts a sample.
    A low perplexity indicates the model is good at predicting the sample.

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

    $predicts = [mx->nd->array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])];
    $labels   = [mx->nd->array([0, 1, 1])];
    $perp = mx->metric->Perplexity(ignore_label=>undef);
    $perp->update($labels, $predicts);
    print $perp->get()
    Perplexity 1.77109762851559

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

=head1 NAME

    AI::MXNet::MAE - Calculate Mean Absolute Error loss
=head1 DESCRIPTION

    >>> $predicts = [mx->nd->array([3, -0.5, 2, 7])->reshape([4,1])]
    >>> $labels = [mx->nd->array([2.5, 0.0, 2, 8])->reshape([4,1])]
    >>> $mean_absolute_error = mx->metric->MAE()
    >>> $mean_absolute_error->update($labels, $predicts)
    >>> print $mean_absolute_error->get()
    ('mae', 0.5)

=cut


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

=head1 NAME

    AI::MXNet::MSE - Calculate Mean Squared Error loss
=head1 DESCRIPTION

    >>> $predicts = [mx->nd->array([3, -0.5, 2, 7])->reshape([4,1])]
    >>> $labels = [mx->nd->array([2.5, 0.0, 2, 8])->reshape([4,1])]
    >>> $mean_squared_error = mx->metric->MSE()
    >>> $mean_squared_error->update($labels, $predicts)
    >>> print $mean_squared_error->get()
    ('mse', 0.375)

=cut

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

=head1 NAME

    AI::MXNet::RMSE - Calculate Root Mean Squred Error loss
=head1 DESCRIPTION

    >>> $predicts = [mx->nd->array([3, -0.5, 2, 7])->reshape([4,1])]
    >>> $labels = [mx->nd->array([2.5, 0.0, 2, 8])->reshape([4,1])]
    >>> $root_mean_squared_error = mx->metric->RMSE()
    >>> $root_mean_squared_error->update($labels, $predicts)
    >>> print $root_mean_squared_error->get()
    'rmse', 0.612372457981

=cut

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


=head1 NAME

    AI::MXNet::CrossEntropy - Calculate Cross Entropy loss
=head1 DESCRIPTION

    >>> $predicts = [mx->nd->array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> $labels   = [mx->nd->array([0, 1, 1])]
    >>> $ce = mx->metric->CrossEntropy()
    >>> $ce->update($labels, $predicts)
    >>> print $ce->get()
    ('cross-entropy', 0.57159948348999023)

=cut

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

=head1 NAME

    AI::MXNet::NegativeLogLikelihood - Computes the negative log-likelihood loss.
=head1 DESCRIPTION

    >>> $predicts = [mx->nd->array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> $labels   = [mx->nd->array([0, 1, 1])]
    >>> $nll_loss = mx->metric->NegativeLogLikelihood
    >>> $nll_loss->update($labels, $predicts)
    >>> print $nll_loss->get()
    ('cross-entropy', 0.57159948348999023)

=cut

package AI::MXNet::NegativeLogLikelihood;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::CrossEntropy';
has '+name'   => (default => 'nll_loss');

package AI::MXNet::PearsonCorrelation;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::EvalMetric';
has '+name'   => (default => 'pearson-correlation');

=head1 NAME

    AI::MXNet::PearsonCorrelation - Computes Pearson correlation.
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

    AI::MXNet::Loss - Dummy metric for directly printing loss.
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

    AI::MXNet::Confidence - Accuracy by confidence buckets.
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

    AI::MXNet::CustomMetric - Custom evaluation metric that takes a sub ref.
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
    nll_loss            AI::MXNet::NegativeLogLikelihood
    f1                  AI::MXNet::F1
    mcc                 AI::MXNet::MCC
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
