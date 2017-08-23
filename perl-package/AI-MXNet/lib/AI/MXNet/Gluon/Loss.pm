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
package AI::MXNet::Gluon::Loss;
use AI::MXNet::Gluon::Block;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Gluon::Loss - Base class for loss.
=cut

=head2 DESCRIPTION

    Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

=head2 _apply_weighting

    Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
=cut


method _apply_weighting(Str $F, GluonInput $loss, Maybe[Num] $weight=, Maybe[AI::MXNet::Symbol] $sample_weight=)
{
    if(defined $sample_weight)
    {
        $loss = $F->broadcast_mul($loss, $sample_weight);
    }
    if(defined $weight)
    {
        $loss = $loss * $weight;
    }
    return $loss;
}

# for symbolic output.shape is not available so we reshape
# to empty shape and let it be inferred from output's shape
# via the '-' operator later.

method _reshape_label_as_output(GluonClass $F, GluonInput $output, GluonInput $label)
{
    if($F eq 'AI::MXNet::NDArray')
    {
        return $label->reshape($output->shape);
    }
    else
    {
        return $label->reshape([]);
    }
}

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'weight'     => (is => 'ro', isa => 'Num');
has 'batch_axis' => (is => 'ro', isa => 'Int');

use overload '""' => sub {
        my $self = shift;
        sprintf(
            "%s(batch_axis=%s, w=%s)",
            $self->_class_name,
            $self->batch_axis,
            $self->weight
        );
    };

method hybrid_forward($F, $x, @args)
{
    confess('NotImplementedError');
}

package AI::MXNet::Gluon::L2Loss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::L2Loss
=cut

=head1 DESCRIPTION

    Calculates the mean squared error between output and label:

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut
has '+weight'     => (default => 1);
has '+batch_axis' => (default => 0);

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[Symbol] :$sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->square($output - $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight/2, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::L1Loss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has '+weight'     => (default => 1);
has '+batch_axis' => (default => 0);

=head1 NAME

    AI::MXNet::Gluon::L1Loss
=cut

=head1 DESCRIPTION

    Calculates the mean absolute error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    Output and label must have the same shape.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[Symbol] :$sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->abs($output - $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'from_sigmoid' => (is => 'ro', isa => 'Bool', default => 0);
has '+batch_axis'  => (default => 0);

=head1 NAME

    AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss
=cut

=head1 DESCRIPTION

    The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression.

    .. math::
        loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))


    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and then BCE, which is more numerically stable through
        log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[Symbol] :$sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss;
    if(not $self->from_sigmoid)
    {
        my $max_val = $F->maximum(-$output, 0);
        $loss = $output - $output*$label + $max_val + $F->log($F->exp(-$max_val)+$F->exp(-$output-$max_val));
    }
    else
    {
        $loss = -($F->log($output+1e-12)*$label + $F->log(1-$output+1e-8)*(1-$label));
    }
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SigmoidBCELoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::SigmoidBinaryCrossEntropyLoss';

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SoftmaxCrossEntropyLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';


=head1 NAME

    AI::MXNet::Gluon::SoftmaxCrossEntropyLoss
=cut

=head1 DESCRIPTION

    Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True`, label should contain integer category indicators:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i {log}(p_{i,{label}_i})

    Label's shape should be output's shape without the `axis` dimension. i.e. for
    `output.shape` = (1,2,3,4) and axis = 2, `label.shape` should be (1,2,4).

    If `sparse_label` is `False`, label should contain probability distribution
    with the same shape as output:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i \\sum_j {label}_j {log}(p_{ij})

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

has 'axis'         => (is => 'ro', isa => 'Int', default => -1);
has '+batch_axis'  => (default => 0);
has 'sparse_label' => (is => 'ro', isa => 'Bool', default => 1);
has 'from_logits'  => (is => 'ro', isa => 'Bool', default => 0);

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[Symbol] :$sample_weight=)
{
    if(not $self->from_logits)
    {
        $output = $F->log_softmax($output);
    }
    my $loss;
    if($self->sparse_label)
    {
        $loss = -$F->pick($output, $label, axis=>$self->axis, keepdims => 1);
    }
    else
    {
        $loss = -$F->sum($output*$label, axis => $self->axis, keepdims => 1);
    }
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SoftmaxCELoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::SoftmaxCrossEntropyLoss';

package AI::MXNet::Gluon::KLDivLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has '+batch_axis'  => (default => 0);
has 'from_logits'  => (is => 'ro', isa => 'Bool', default => 1);

=head1 NAME

    AI::MXNet::Gluon::KLDivLoss
=cut

=head1 DESCRIPTION

    The Kullback-Leibler divergence loss.

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    .. math::
        L = 1/n \\sum_i (label_i * (log(label_i) - output_i))

    Label's shape should be the same as output's.

    Parameters
    ----------
    from_logits : bool, default is `True`
        Whether the input is log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[Symbol] :$sample_weight=)
{
    if(not $self->from_logits)
    {
        $output = $F->log_softmax($output);
    }
    my $loss = $label * ($F->log($label+1e-12) - $output);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

1;