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

package AI::MXNet::LRScheduler;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Logging;
use overload "&{}" => sub { my $self = shift; sub { $self->call(@_) } },
             fallback => 1;

=head1 NAME

    AI::MXNet::LRScheduler - The adaptive scheduler of the learning rate.
=cut

=head1 DESCRIPTION

    Learning rate scheduler, which adaptively changes the learning rate based on the
    progress.
=cut

=head2 new

    base_lr : float (optional, default 0.01)
    the initial learning rate
=cut

has 'base_lr' => (is => 'rw', isa => 'Num', default => 0.01);

=head2 call

    Call to schedule current learning rate

    The training progress is presented by num_update, which can be roughly
    viewed as the number of minibatches executed so far. Its value is
    non-decreasing, and increases at most by one.

    The exact value is the upper bound of the number of updates applied to
    a weight/index

    See more details in https://github.com/dmlc/mxnet/issues/625

    Parameters
    ----------
    num_update: int
        the maximal number of updates applied to a weight.
=cut

package AI::MXNet::FactorScheduler;

=head1 NAME

    AI::MXNet::FactorScheduler - Reduces the learning rate by a factor.

=head1 DESCRIPTION

    Reduces the learning rate by a factor each step.
    Assume the weight has been updated by n times, then the learning rate will
    be base_lr * factor^(floor(n/step))

    Parameters
    ----------
    step: int
        schedule the learning rate update after n updates
    factor: float
        the factor by which to reduce the learning rate.
=cut
use Mouse;
extends 'AI::MXNet::LRScheduler';

has 'step'            => (is => 'ro', isa => 'Int', required => 1);
has 'factor'          => (is => 'ro', isa => 'Num', default  => 1);
has 'count'           => (is => 'rw', isa => 'Int', default  => 1);
has 'stop_factor_lr'  => (is => 'ro', isa => 'Num', default  => 1e-8);

sub BUILD
{
    my $self = shift;
    confess("Schedule step must be greater or equal than 1")
        if $self->step < 1;
    confess("Factor must be no more than 1 to make lr reduce")
        if $self->factor > 1;
}

method call(Int $num_update)
{
    # NOTE: use while rather than if  (for continuing training via load_epoch)
    while($num_update > $self->count + $self->step)
    {
        $self->count($self->count + $self->step);
        $self->base_lr($self->base_lr * $self->factor);
        if($self->base_lr < $self->stop_factor_lr)
        {
            $self->base_lr($self->stop_factor_lr);
            AI::MXNet::Logging->info(
                "Update[%d]: now learning rate arrived at %0.5e, will not "
                ."change in the future", $num_update, $self->base_lr
            );
        }
        else
        {
            AI::MXNet::Logging->info(
                "Update[%d]: Changed learning rate to %0.5e",
                $num_update, $self->base_lr
            );
        }
    }
    return $self->base_lr;
}

package AI::MXNet::MultiFactorScheduler;

=head1 NAME

    AI::MXNet::MultiFactorScheduler - Reduces the learning rate by an array ref of factors.

=head1 DESCRIPTION

    Reduces a learning rate in factor at steps specified in an array ref.
    Assume the weight has been updated by n times, then the learning rate will
    be base_lr * factor^(sum((step/n)<=1)) # step is an array.

    Parameters
    ----------
    step: array ref of int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
=cut

use Mouse;
extends 'AI::MXNet::LRScheduler';
has 'step'            => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has 'factor'          => (is => 'ro', isa => 'Num', default  => 1);
has 'cur_step_ind'    => (is => 'rw', isa => 'Int', default  => 0);
has 'count'           => (is => 'rw', isa => 'Int', default  => 0);

sub BUILD
{
    my $self = shift;
    confess("step array must have at least one member")
        unless @{ $self->step } >=1 ;
    for (my $i = 0; $i < @{ $self->step }; $i++)
    {
        confess("Schedule step must be an increasing integer list")
            if($i and $self->step->[$i] <= $self->step->[$i-1]);
        confess("Schedule step must be greater or equal than 1")
            if $self->step->[$i] < 1;
    }
    confess("Factor must be no more than 1 to make lr reduce")
        if $self->factor > 1;
}

method call(Int $num_update)
{
    # NOTE: use while rather than if  (for continuing training via load_epoch)
    while($self->cur_step_ind < @{ $self->step })
    {
        if($num_update > $self->step->[$self->cur_step_ind])
        {
            $self->count($self->step->[$self->cur_step_ind]);
            $self->cur_step_ind($self->cur_step_ind + 1);
            $self->base_lr($self->base_lr * $self->factor);
            AI::MXNet::Logging->info(
                "Update[%d]: Changed learning rate to %0.5e",
                $num_update, $self->base_lr
            );
        }
        else
        {
            return $self->base_lr;
        }
    }
    return $self->base_lr;
}

1;
