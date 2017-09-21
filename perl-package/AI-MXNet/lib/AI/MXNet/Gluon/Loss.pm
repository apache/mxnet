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


method _apply_weighting(Str $F, GluonInput $loss, Maybe[Num] $weight=, Maybe[GluonInput] $sample_weight=)
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
        return $label->reshape(shape => []);
    }
}

use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'weight'     => (is => 'rw', isa => 'Num');
has 'batch_axis' => (is => 'rw', isa => 'Int', default => 0);

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss;
    if(not $self->from_sigmoid)
    {
        my $max_val = (-$output)->maximum(0);
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
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

__PACKAGE__->register('AI::MXNet::Gluon::Loss');


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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
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

package AI::MXNet::Gluon::CTCLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'layout'        => (is => 'rw', isa => 'Str', default => 'NTC');
has 'label_layout'  => (is => 'rw', isa => 'Str', default => 'NT');

=head1 NAME

    AI::MXNet::Gluon::CTCLoss
=cut

=head1 DESCRIPTION

    Connectionist Temporal Classification Loss.

    See `"Connectionist Temporal Classification: Labelling Unsegmented
    Sequence Data with Recurrent Neural Networks"
    <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ paper for more information.

    Parameters
    ----------
    layout : str, default 'NTC'
        Layout of the output sequence activation vector.
    label_layout : str, default 'NT'
        Layout of the labels.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
        This should be used as the fifth argument when calling this loss.

    Input shapes:
        `data` is an activation tensor (i.e. before softmax).
        Its shape depends on `layout`. For `layout='TNC'`, this
        input has shape `(sequence_length, batch_size, alphabet_size)`
        Note that the last dimension with index `alphabet_size-1` is reserved for special
        blank character.

        `label` is the label index matrix with zero-indexed labels.
        Its shape depends on `label_layout`. For `label_layout='TN'`, this
        input has shape `(label_sequence_length, batch_size)`. Padding mask of value ``-1``
        is available for dealing with unaligned label lengths.
        When `label_lengths` is specified, label lengths are directly used and padding mask
        is not allowed in the label.
        When `label_lengths` is not specified, the first occurrence of ``-1``
        in each sample marks the end of the label sequence of that sample.

        For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have three
        sequences 'ba', 'cbb', and 'abac'. We can index the labels as `{'a': 0, 'b': 1, 'c': 2}`.
        The alphabet size should be 4, and we reserve the channel index 3 for blank label
        in data tensor. The padding mask value for extra length is -1, so the resulting `label`
        tensor should be padded to be::

          [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]

        `data_lengths` is optional and defaults to None.
        When specified, it represents the actual lengths of data.
        The shape should be (batch_size,).
        If None, the data lengths are treated as being equal to the max sequence length.
        This should be used as the third argument when calling this loss.

        `label_lengths` is optional and defaults to None.
        When specified, it represents the actual lengths of labels.
        The shape should be (batch_size,).
        If None, the label lengths are derived from the first occurrence of
        the value specified by `padding_mask`.
        This should be used as the fourth argument when calling this loss.

    Output shape:
        The CTC loss output has the shape (batch_size,).
=cut
use AI::MXNet::Base;

sub BUILD
{
    my $self = shift;
    assert(
        (grep { $_ eq $self->layout } ('NTC', 'TNC')),\
        "Only 'NTC' and 'TNC' layouts for output are supported. Got: ${\ $self->layout }"
    );
    assert(
        (grep { $_ eq $self->label_layout } ('NT', 'TN')),\
        "Only 'NT' and 'TN' layouts for label are supported. Got: ${\ $self->label_layout }"
    );
    $self->batch_axis(index($self->label_layout, 'N'));
}

method hybrid_forward(
    GluonClass $F, GluonInput $data, GluonInput $label,
    Maybe[GluonInput] $data_lengths=, Maybe[GluonInput] $label_lengths=, Maybe[GluonInput] $sample_weight=
)
{
    if($self->layout eq 'NTC')
    {
        $data = $F->swapaxes($data, dim1 => 0, dim2 => 1);
    }
    if($self->batch_axis == 1)
    {
        $label = $F->swapaxes($label, dim1 => 0, dim2 => 1);
    }
    my $loss = $F->contrib->CTCLoss(
        $data, $label,
        (defined $data_lengths ? $data_lengths : ()),
        (defined $label_lengths ? $label_lengths : ()),
        use_data_lengths  => defined $data_lengths ? 1 : 0,
        use_label_lengths => defined $label_lengths ? 1 : 0,
        blank_label=>'last'
    );
    return $self->_apply_weighting($F, $loss, $self->weight, $sample_weight);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');


package AI::MXNet::Gluon::Huber;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'rho'        => (is => 'rw', isa => 'Num', default => 1);

=head1 NAME

    AI::MXNet::Gluon::Huber
=cut

=head1 DESCRIPTION

    Calculates Huber's robust loss function yielding a trimmed mean estimator, i.e.
       L2 loss in the center and L1 loss for deviations beyond rho:

    .. math::
        L = \\begin{cases} \\frac{1}{2 \\rho} ({output}_i - {label}_i)^2 &
                           \\text{ if } |{output}_i - {label}_i| < \\rho \\\
                           |{output}_i - {label}_i| - \\frac{\\rho}{2} &
                           \\text{ otherwise }
            \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    rho : float
        Threshold for trimmed mean estimator. By default set to 1
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->abs($output - $label);
    $loss = (($loss > $self->rho) * ($loss - 0.5 * $self->rho) +
                (0.5/$self->rho) * ($loss <= $self->rho) * $loss**2);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::EpsilonInsensitive;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'epsilon'        => (is => 'rw', isa => 'Num', default => 0.1);

=head1 NAME

    AI::MXNet::Gluon::EpsilonInsensitive
=cut

=head1 DESCRIPTION

    Calculates Huber's robust loss function yielding a trimmed mean estimator, i.e.
       L2 loss in the center and L1 loss for deviations beyond rho:

    .. math::
        L = \\mathrm{max}(0, |{output}_i - {label}_i| - \\epsilon)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    epsilon : float
        Threshold for trimmed epsilon-insensitivity parameter. By default set to 0.1
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = ($F->abs($output - $label) - $self->epsilon)->maximum($F->zeros_like($output));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SoftMargin;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::SoftMargin
=cut

=head1 DESCRIPTION

    Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = max(0, 1 - {output}_i {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = (1.0 - $output * $label)->maximum($F->zeros_like($output));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::SquaredSoftMargin;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::SquaredSoftMargin
=cut

=head1 DESCRIPTION

    Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = max(0, 1 - {output}_i {label}_i)^2

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = ((1.0 - $output * $label)->maximum($F->zeros_like($output)))**2;
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::Exponential;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::Exponential
=cut

=head1 DESCRIPTION

    Calculates the exponential hinge loss (quite obscure):

    .. math::
        L = \\exp(- {output}_i {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->exp(-$output * $label);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::Logistic;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::Logistic
=cut

=head1 DESCRIPTION

    Calculates the logistic loss (for binary losses only):

    .. math::
        L = \\log(1 + \\exp(- {output}_i {label}_i))

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->log(1 + $F->exp(-$output * $label));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::Quantile;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'tau' => (is => 'rw', isa => 'Num', default => 0.5);

=head1 NAME

    AI::MXNet::Gluon::Quantile
=cut

=head1 DESCRIPTION

    Calculates Koenker's quantile regression loss function yielding an estimate of the
       appropriately chosen quantile rather than the mean (or median):

    .. math::
        L = {max}(\\tau ({output}_i - {label}_i), (1-\\tau) ({label}_i - {output}_i)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    tau : float
        Quantile of the estimator. By default set to 0.5, i.e. by
        default identical with L1 loss, up to a scaling factor.
        This must be in the range (0,1).
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $output - $label;
    $loss = ($self->tau * $loss)->maximum(($self->tau - 1) * $loss);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::Langford;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::Langford
=cut

=head1 DESCRIPTION

    Calculates the Huberized soft-margin loss that is used in VW (Vowpal Wabbit).
       It is given by a squared loss for margin values of [-1, 0] and by a linear
       loss for values larger than that.

    .. math::
        L = \\begin{cases}
          0 &
          \\text{ if } {output}_i {label}_i > 1 \\\
          \\frac{1}{2} - {output}_i {label}_i &
          \\text{ if } {output}_i {label}_i < 0 \\\
          \\frac{1}{2} (1 - {output}_i {label}_i)^2 &
          \\text{ otherwise }
          \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->zeros_like($output)->maximum(1 - $output * $label);
    $loss = ($loss < 1.0) * 0.5 * ($loss**2) + ($loss >= 1.0) * ($loss - 0.5);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::DualKL;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::DualKL
=cut

=head1 DESCRIPTION

    Estimates the Kullback Leibler Divergence between two
       distributions by convex duality. See Nguyen, Wainwright and
       Jordan (NGW), 2008 for a detailed derivation. In a nutshell it
       estimates:

    .. math::
       KL(p\\|q) = E_p[\\log p(x)] - E_p[\\log q(x)]

       Clearly this isn't easy to compute. Hence, NGW use the dual of
       the F-divergence log p(x)/q(x) and pose it as an optimization
       problem. This leads to the following loss function, which is
       different for both distributions (which we treat as a binary
       classification problem). The function that is being estimated
       allows us to get the Radon-Nikodym via dp/dq = exp(f).

    .. math::
        L = \\begin{cases}
            \\exp(f) & \\text{ if } y = -1 \\\
             -f-1 & \text{ if } y = 1
          \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = ($label == -1) * $F->exp($output) - ($label == 1) * ($output + 1);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::RelativeNovelty;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'rho' => (is => 'rw', isa => 'Num', default => 0.1);

=head1 NAME

    AI::MXNet::Gluon::RelativeNovelty
=cut

=head1 DESCRIPTION

    Estimates a relative novelty detector. See the Song, Teo and
       Smola (STS), 2009 for details. The main point is to estimate
       the ratio dp/dq well via max(0, rho - log dp/dq). As with the
       KL divergence estimator, the Fenchel-Legendre dual is easier to
       deal with. This leads to the following loss function:

    .. math::
        L = \\begin{cases}
            \\exp(f - rho) & \\text{ if } y = -1 \\\
             -f-1 & \\text{ if } y = 1 \\text{ and } f > 0 \\
            \\exp(f) & \\text{ if } y = 1 \\text{ and } f <= 0
          \\end{cases}

    output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    rho : float
        Relative probability weight for the most prevalent part of the
        probability distribution. It needs to be (0, 1). Defaults to 0.1
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = -($output > 0) * ($output + 1) - ($output <= 0) * $F->exp($output);
    $loss = ($label == 1) * $loss + ($label == -1) * $F->exp($output - $self->rho);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::LogCosh;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::LogCosh
=cut

=head1 DESCRIPTION

    Calculates the smoothed L1 loss, aka log cosh loss in a
       numerically stable manner (i.e. without exponentiating large
       values of the cosh function.

    .. math::
        L = \\log 2 \\cosh ({output}_i - {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->abs($label - $output);
    $loss = $loss + $F->log(0.5 + 0.5 * $F->exp(-2 * $loss));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::Poisson;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';

=head1 NAME

    AI::MXNet::Gluon::Poisson
=cut

=head1 DESCRIPTION

    Calculates the Poisson loss function (up to the normalization
       by a factorial in the label, due to computational efficiency
       reasons).

       NOTE THAT THIS IS DIFFERENT FROM THE POISSON LOSS IN PYTORCH
       AND KERAS INSOFAR AS IT USES THE EPXONENTIAL VERSION. THAT ONE
       DOESN'T SUFFER FROM LOG 0 PROBLEMS.

    .. math::
        L = -\\log p({label}_i|{output}_i)
          = \\log {label}_i! + \\exp({output}_i) - {output}_i {label}_i

    Output and label must have the same shape. This is a scalar loss function.

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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    $label = __PACKAGE__->_reshape_label_as_output($F, $output, $label);
    my $loss = $F->exp($output) - $output * $label;
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::MaxMargin;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'axis'  => (is => 'rw', isa => 'Int', default => -1);
has 'delta' => (is => 'rw', init_arg => undef);

=head1 NAME

    AI::MXNet::Gluon::MaxMargin
=cut

=head1 DESCRIPTION

    Calculates the MaxMargin loss, aka multiclass soft-margin
       loss. This requires access to a multiclass loss matrix delta
       which measures the cost for misclassifying label y as y'. This
       matrix can be specified at construction time. If it does not
       exist, we will susbstitute it with a 0-1 loss with automagic
       size inference.

    .. math::
       L = {max}_{y} [\\delta({label}, y) + {output}[y]] - {output}_{label}

    Label's shape should be output's shape without the `axis` dimension. i.e. for
    `output.shape` = (1,2,3,4) and axis = 2, `label.shape` should be (1,2,4).

    Parameters
    ----------
    delta : loss matrix, default None. In this case it is presumed to
        be a (0,1) loss, i.e. a constant loss of 1 for all
        misclassifications. Otherwise its dimensionality must match
        that of the number of classes.
    axis : int, default -1
        The axis to sum over when taking the maximum.
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

method hybrid_forward(GluonClass $F, GluonInput $output, GluonInput $label, Maybe[GluonInput] $sample_weight=)
{
    # Check if the cost matrix has been defined. If not, use a
    # dumb (0,1) loss. This is only executed once, the first time
    # you invoke the loss.
    if (not defined $self->delta)
    {
        my $classes = $output->shape->[$self->axis];
        $self->delta($F->ones(shape=>[$classes, $classes]));
        for my $i (0..$classes-1)
        {
            $self->delta->slice($i, $i) .= 0;
        }
    }
    my $loss = -$F->pick($output, $label, axis=>$self->axis, keepdims=>1);
    $loss += $F->max($output + $F->take($self->delta, $label), axis=>$self->axis, keepdims=>1);
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

package AI::MXNet::Gluon::TripletLoss;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::Loss';
has 'margin'  => (is => 'rw', isa => 'Num', default => 1);
has 'axis'    => (is => 'rw', isa => 'Int', default => 1);

=head1 NAME

    AI::MXNet::Gluon::TripletLoss
=cut

=head1 DESCRIPTION

    Calculates the mean squared error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert^2.

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    margin : float
        Margin of separation between correct and incorrect pair. Defaults to 1.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    axis : int, default 1
        The axis over which to sum distances.
    batch_axis : int, default 0
        The axis that represents mini-batch.
=cut

method hybrid_forward(
    GluonClass $F,
    GluonInput $output1, GluonInput $output2, GluonInput $output3,
    Maybe[GluonInput] $sample_weight=
)
{
    my $loss = $F->sum(($output1-$output2)**2 - ($output1-$output3)**2, axis=>self->axis) + $self->margin;
    $loss = $loss->maximum($F->zeros_like($loss));
    $loss = __PACKAGE__->_apply_weighting($F, $loss, $self->weight, $sample_weight);
    return $F->mean($loss, axis => $self->batch_axis, exclude => 1);
}

__PACKAGE__->register('AI::MXNet::Gluon::Loss');

1;