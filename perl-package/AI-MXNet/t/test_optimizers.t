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

package PerlAdam;
use strict;
use warnings;
use AI::MXNet qw(mx);
use Mouse;
use AI::MXNet::Function::Parameters;
extends 'AI::MXNet::Optimizer';
has 'beta1' => (is => 'rw', default => 0.9);
has 'beta2' => (is => 'rw', default => 0.999);
has 'epsilon' => (is => 'rw', default => 1e-8);
has 'rescale_grad' => (is => 'rw', default => 1);
has 'decay_factor' => (is => 'rw', default => (1-1e-8));
around BUILDARGS => \&init;

func init($code, $class, %kwargs)
{
    return $class->$code(learning_rate => 0.001, wd => 0.9, %kwargs);
}

=begin
        Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data
=cut

method create_state($index, $weight)
{
    return [
            mx->nd->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype),  # mean
            mx->nd->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype)   # variance
    ]; 
}

=begin
        Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
=cut

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    $self->_update_count($index);
    my $t = $self->_index_update_count->{$index};
    my ($mean, $variance) = @$state;
    my $wd = $self->_get_wd($index);
    $grad = $grad * $self->rescale_grad + $wd * $weight;
    if($self->clip_gradient)
    {
        mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient, out => $grad);
    }
    $mean *= $self->beta1;
    $mean += $grad * (1 - $self->beta1);

    $variance *= $self->beta2;
    $variance += (1 - $self->beta2) * mx->nd->square($grad, out => $grad);

    my $coef1 = 1 - $self->beta1**$t;
    my $coef2 = 1 - $self->beta2**$t;
    $lr *= sqrt($coef2)/$coef1;
    $weight -= $lr*$mean/(mx->nd->sqrt($variance) + $self->epsilon);
}

=head

    RMSProp optimizer of Tieleman & Hinton, 2012,

    For centered=False, the code follows the version in
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012

    For centered=True, the code follows the version in
    http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.001.
    gamma1: float, optional
        decay factor of moving average for gradient, gradient^2.
        Default value is set to 0.9.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
        Only used if centered=True
    epsilon : float, optional
        Default value is set to 1e-8.
    centered : boolean, optional
        Use Graves or Tielemans & Hintons version of RMSProp
    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    clip_weights : float, optional
        clip weights in range [-clip_weights, clip_weights]
=cut

package PerlRMSProp;
use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.001);
has 'gamma1'         => (is => "ro", isa => "Num",  default => 0.9);
has 'gamma2'         => (is => "ro", isa => "Num",  default => 0.9);
has 'epsilon'        => (is => "ro", isa => "Num",  default => 1e-8);
has 'centered'       => (is => "ro", isa => "Bool", default => 0);
has 'clip_weights'   => (is => "ro", isa => "Num");

# For centered=False: n
# For centered=True: n, g, delta
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            $self->centered
            ? (
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # n
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # g
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                )
            )   # delta
            : (
                AI::MXNet::NDArray->zeros(
                    $weight->shape,
                    ctx => $weight->context
                ),  # n
            )
    ];
}

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad = $grad * $self->rescale_grad + $wd * $weight;
    if(not $self->centered)
    {
        my ($n) = @$state;
        if(defined $self->clip_gradient)
        {
            $grad = mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient);
        }
        $n .= (1 - $self->gamma1) * ($grad * $grad) + $self->gamma1 * $n;
        $weight -= $lr * $grad/(mx->nd->sqrt($n + $self->epsilon));
    }
    else
    {
        my ($n, $g, $delta) = @$state;
        if(defined $self->clip_gradient)
        {
            $grad = mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient);
        }
        $n .= (1 - $self->gamma1) * ($grad * $grad) + $self->gamma1 * $n;
        $g .= (1 - $self->gamma1) * $grad + $self->gamma1 * $g;
        $delta .= ($self->gamma2) * $delta - $lr * $grad/(mx->nd->sqrt($n - $g*$g + $self->epsilon));
        $weight += $delta;
    }
    if($self->clip_weights)
    {
        mx->nd->clip($weight, -$self->clip_weights, $self->clip_weights, out => $weight);
    }
}

package PerlSGD;
# perl reference implemenation of sgd
use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.01);
has 'momentum'       => (is => "ro", isa => "Num",  default => 0);
has 'multi_precision' => (is => 'ro', isa => 'Bool', default => 0);

# Create additional optimizer state: momentum
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    my $momentum;
    my $weight_master_copy;
    my $do_multi_precision = ($self->multi_precision and $weight->dtype eq 'float16');
    if($do_multi_precision)
    {
        if($self->momentum != 0)
        {
            $momentum = mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>'float32');
        }
        $weight_master_copy = mx->nd->array($weight, ctx=>$weight->context, dtype=>'float32');
        return [$momentum, $weight_master_copy];
    }
    else
    {
        if($self->momentum != 0)
        {
            $momentum = mx->nd->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype);
        }
    }
    return $momentum;
}

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $use_multi_precision = ref($state) eq 'ARRAY';

    if(not $use_multi_precision)
    {
        if($self->momentum == 0)
        {
            if(defined $self->clip_gradient)
            {
                $weight .= ((1 - $lr*$wd)*$weight -
                    $lr * mx->nd->clip($grad*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
                );
            }
            else
            {
                $weight .= (1 - $lr*$wd)*$weight - $lr*$self->rescale_grad*$grad;
            }
        }
        else
        {
            my $mom = $state;
            if(defined $self->clip_gradient)
            {
                $mom .= ($self->momentum*$mom - $lr*$wd*$weight -
                    $lr * mx->nd->clip($grad*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
                );
                $weight += $mom;
            }
            else
            {
                $mom .= $self->momentum*$mom - $lr*$wd*$weight - $lr*$self->rescale_grad*$grad;
                $weight += $mom;
            }
        }
    }
    else
    {
        my $grad32 = mx->nd->array($grad, ctx=>$grad->context, dtype=>'float32');
        my $mom = $state->[0];
        my $weight32 = $state->[1];
        if($self->momentum == 0)
        {
            if(defined $self->clip_gradient)
            {
                $weight32 .= ((1 - $lr*$wd)*$weight32 -
                    $lr * mx->nd->clip($grad32*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
                );
            }
            else
            {
                $weight32 .= (1 - $lr*$wd)*$weight32 - $lr*$self->rescale_grad*$grad32;
            }
        }
        else
        {
            if(defined $self->clip_gradient)
            {
                $mom .= ($self->momentum*$mom - $lr*$wd*$weight32 -
                    $lr * mx->nd->clip($grad32*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
                );
                $weight32 += $mom;
            }
            else
            {
                $mom .= $self->momentum*$mom - $lr*$wd*$weight32 - $lr*$self->rescale_grad*$grad32;
                $weight32 += $mom;
            }
        }
        my $tmp = $weight32->astype($weight->dtype);
        $tmp->copyto($weight);
    }
}

package PerlSparseSGD;
# perl reference implemenation of sgd
use Mouse;
use AI::MXNet::TestUtils qw(almost_equal);
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.01);
has 'momentum'       => (is => "ro", isa => "Num",  default => 0);
has 'multi_precision' => (is => 'ro', isa => 'Bool', default => 0);
has 'lazy_update' => (is => 'ro', isa => 'Bool', default => 1);

method create_state($index, $weight)
{
    if($self->momentum == 0)
    {
        return undef;
    }
    else
    {
        return mx->nd->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype);
    }
}

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $num_rows = $weight->shape->[0];
    if($self->momentum == 0)
    {
        # Update on a per row basis, skip all-zero rows
        for my $row (0..$num_rows-1)
        {
            my $grad_row = $grad->at($row);
            my $all_zeros = almost_equal($grad_row->aspdl, mx->nd->zeros($grad_row->shape, ctx => $grad_row->context, dtype => $grad_row->dtype)->aspdl);
            if($all_zeros and $self->lazy_update)
            {
                next;
            }
            if(defined $self->clip_gradient)
            {
                $weight->at($row) .= (
                    (1 - $lr*$wd)*$weight->at($row) -
                    $lr * mx->nd->clip(
                        $grad->at($row)*$self->rescale_grad,
                        -$self->clip_gradient, $self->clip_gradient
                    )
                );
            }
            else
            {
                $weight->at($row) .= (1 - $lr*$wd)*$weight->at($row) - $lr*$self->rescale_grad*$grad->at($row);
            }
        }
    }
    else
    {
        my $mom = $state;
        for my $row (0..$num_rows-1)
        {
            my $grad_row = $grad->at($row);
            my $all_zeros = almost_equal($grad_row->aspdl, mx->nd->zeros($grad_row->shape, ctx => $grad_row->context, dtype => $grad_row->dtype)->aspdl);
            if($all_zeros and $self->lazy_update)
            {
                next;
            }
            if(defined $self->clip_gradient)
            {
                $mom->at($row) .= ($self->momentum*$mom->at($row) - $lr*$wd*$weight->at($row) -
                    $lr * mx->nd->clip($grad->at($row)*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
                );
                $weight->at($row) += $mom->at($row);
            }
            else
            {
                $mom->at($row) .= $self->momentum*$mom->at($row) - $lr*$wd*$weight->at($row) - $lr*$self->rescale_grad*$grad->at($row);
                $weight->at($row) += $mom->at($row);
            }
        }
    }

}

package PerlNAG;
use Mouse;
extends 'PerlSGD';

method create_state($index, $weight)
{
    my $momentum;
    my $weight_master_copy;
    my $do_multi_precision = ($self->multi_precision and $weight->dtype eq 'float16');
    if($do_multi_precision)
    {
        if($self->momentum != 0)
        {
            $momentum = mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>'float32');
        }
        $weight_master_copy = mx->nd->array($weight, ctx=>$weight->context, dtype=>'float32');
        return [$weight_master_copy, $momentum];
    }
    else
    {
        if($self->momentum != 0)
        {
            $momentum = mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype);
        }
        return $momentum;
    }
}

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $use_multi_precision = (defined $state and not Scalar::Util::blessed($state) and ref($state eq 'ARRAY'));
    if(not $use_multi_precision)
    {
        $grad *= $self->rescale_grad;
        if(defined $self->clip_gradient)
        {
            $grad = mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient);
        }
        if($self->momentum == 0)
        {
            $weight += -$lr * ($grad + $wd * $weight);
        }
        else
        {
            my $mom = $state;
            $mom *= $self->momentum;
            $grad += $wd * $weight;
            $mom += $grad;
            $grad += $self->momentum * $mom;
            $weight += -$lr * $grad;
        }
    }
    else
    {
        my $grad32 = mx->nd->array($grad, ctx=>$grad->context, dtype=>'float32');
        $grad32 *= $self->rescale_grad;
        if(defined $self->clip_gradient)
        {
            $grad32 = mx->nd->clip($grad32, -$self->clip_gradient, $self->clip_gradient);
        }
        my $mom = $state->[1];
        my $weight32 = $state->[0];
        if($self->momentum == 0)
        {
            $weight32 += -$lr * ($grad32 + $wd * $weight32);
        }
        else
        {
            $mom *= $self->momentum;
            $grad32 += $wd * $weight32;
            $mom += $grad32;
            $grad32 += $self->momentum * $mom;
            $weight32 += -$lr * $grad32;
        }
        my $tmp = $weight32->astype($weight->dtype);
        $tmp->copyto($weight);
    }
}

package PerlFTML;
use Mouse;
extends 'AI::MXNet::Optimizer';
has 'beta1' => (is => 'rw', default => 0.6);
has 'beta2' => (is => 'rw', default => 0.999);
has 'epsilon' => (is => 'rw', default => 1e-8);

method create_state($index, $weight)
{
    return [mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype), # d_0
            mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype), # v_0
            mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype)] # z_0
}

method update($index, $weight, $grad, $state)
{
    $self->_update_count($index);
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    my $t = $self->_index_update_count->{$index};

    my $grad = $grad * $self->rescale_grad + $wd * $weight;
    if(defined $self->clip_gradient)
    {
        $grad = mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient);
    }
    # get previous states
    my ($prev_d, $prev_v, $prev_z) = @{ $state };
    # compute states
    my $v_t = $self->beta2 * $prev_v + (1 - $self->beta2) * mx->nd->square($grad);
    my $d_t = (1 - ($self->beta1**$t)) / $lr * (mx->nd->sqrt($v_t / (1 - ($self->beta2**$t))) + $self->epsilon);
    my $sigma_t = $d_t - $self->beta1 * $prev_d;
    my $z_t = $self->beta1 * $prev_z + (1 - $self->beta1) * $grad - $sigma_t * $weight;
    # update weight
    $weight .= - $z_t / $d_t;
    # update states
    $prev_d .= $d_t;
    $prev_v .= $v_t;
    $prev_z .= $z_t;
}

package PerlSignum;
use Mouse;
extends 'AI::MXNet::Optimizer';
has 'wd_lh' => (is => 'rw', default => 0);
has 'momentum' => (is => 'rw', default => 0.9);

method create_state($index, $weight)
{
    if($self->momentum != 0)
    {
        return mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype, stype=>$weight->stype);
    }
    return undef;
}

method update($index, $weight, $grad, $state)
{
    $self->_update_count($index);
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    if(defined $state)
    {
        my $mom = $state;
        if(defined $self->clip_gradient)
        {
            $mom .= ($self->momentum*$mom - (1-$self->momentum)*($wd*$weight +
                mx->nd->clip($grad*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)));
        }
        else
        {
            $mom .= $self->momentum*$mom - (1-$self->momentum)*$wd*$weight - (1-$self->momentum)*$self->rescale_grad*$grad;
        }
        $weight .= (1 - $lr*$self->wd_lh)*$weight + $lr * mx->nd->sign($mom);
    }
    else
    {
        $weight .= (1 - $lr*($wd+$self->wd_lh))*$weight - $lr * mx->nd->sign($grad);
    }
}

package PerlFtrl;
use Mouse;
use AI::MXNet::TestUtils qw(almost_equal);
extends 'AI::MXNet::Optimizer';

has 'lamda1' => (is => 'rw', default => 0.01);
has '+learning_rate' => (default => 0.1);
has 'beta' => (is => 'rw', default => 1);
has 'sparse_update' => (is => 'rw', default => 0);

method create_state($index, $weight)
{
    return [
        mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype),  # dn
        mx->nd->zeros($weight->shape, ctx => $weight->context, dtype=>$weight->dtype)   # n
    ];
}

method update($index, $weight, $grad, $state)
{
    $self->_update_count($index);
    my $wd = $self->_get_wd($index);
    my $lr = $self->_get_lr($index);
    my $num_rows = $weight->shape->[0];

    my ($dn, $n) = @$state;
    for my $row (0..$num_rows-1)
    {
        my $grad_row = $grad->at($row);
        my $all_zeros = almost_equal($grad_row->aspdl, mx->nd->zeros($grad_row->shape, ctx => $grad_row->context, dtype => $grad_row->dtype)->aspdl);
        if($all_zeros and $self->sparse_update)
        {
            next;
        }
        $grad_row *= $self->rescale_grad;
        if(defined $self->clip_gradient)
        {
            $grad_row .= mx->nd->clip($grad_row, -$self->clip_gradient, $self->clip_gradient);
        }

        #update dn, n
        $dn->at($row) += $grad_row - (mx->nd->sqrt($n->at($row) + $grad_row * $grad_row) - mx->nd->sqrt($n->at($row))) * $weight->at($row) / $lr;
        $n->at($row) += $grad_row * $grad_row;

        # update weight
        $weight->at($row) .= (mx->nd->sign($dn->at($row)) * $self->lamda1 - $dn->at($row)) /
                          (($self->beta + mx->nd->sqrt($n->at($row))) / $lr + $wd) * (mx->nd->abs($dn->at($row)) > $self->lamda1);
    }
}

package PerlAdaGrad;
use Mouse;
extends 'AI::MXNet::Optimizer';

has 'eps' => (is => 'rw', default => 1e-7);
method create_state($index, $weight)
{
    mx->nd->zeros($weight->shape, ctx => $weight->context, stype => $weight->stype);
}

method update($index, $weight, $grad, $state)
{
    $self->_update_count($index);
    my $wd = $self->_get_wd($index);
    my $lr = $self->_get_lr($index);
    my $num_rows = $weight->shape->[0];
    my $history = $state;
    $grad *= $self->rescale_grad;
    if(defined $self->clip_gradient)
    {
        $grad = mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient);
    }
    $history += mx->nd->square($grad);
    my $div = $grad / mx->nd->sqrt($history + $self->eps);
    $weight += ($div + $weight * $wd) * -$lr;
}

package main;
use Carp;
use Test::More tests => 7884;
use AI::MXNet::Base;
use PDL::NiceSlice;
use AI::MXNet::TestUtils qw(same reldiff almost_equal rand_ndarray);
use AI::MXNet::Function::Parameters;

func compare_optimizer($opt1, $opt2, $shape, $dtype, $w_stype='default', $g_stype='default')
{
    my ($w1, $w2, $g1, $g2);
    if($w_stype eq 'default')
    {
        $w1 = mx->random->uniform({shape => $shape, ctx => mx->cpu, dtype=>$dtype});
        $w2 = $w1->copyto(mx->cpu());
    }
    elsif($w_stype eq 'row_sparse' or $w_stype eq 'csr')
    {
        $w2 = rand_ndarray($shape, $w_stype, 1, $dtype);
        $w1 = $w2->copyto(mx->cpu())->tostype('default');
    }
    else
    {
        Carp::confess("type not supported yet");
    }
    if($g_stype eq 'default')
    {
        $g2 = mx->random->uniform(shape=>$shape, ctx=>mx->cpu, dtype=>$dtype);
        $g1 = $g2->copyto(mx->cpu);
    }
    elsif($g_stype eq 'row_sparse' or $g_stype eq 'csr')
    {
        $g2 = rand_ndarray($shape, $g_stype, rand(), $dtype);
        $g1 = $g2->copyto(mx->cpu)->tostype('default');
    }
    else
    {
        Carp::confess("type not supported yet");
    }

    my $state1 = $opt1->create_state(0, $w1);
    my $state2 = $opt2->create_state(0, $w2);
    zip(
        sub {
            my ($s1, $s2) = @_;
            ok(same($s1->aspdl, $s2->aspdl)) if defined $s1 and defined $s2;
        },
        ref $state1 eq 'ARRAY' ? $state1 : [$state1], ref $state2 eq 'ARRAY' ? $state2 : [$state2]
    ) if defined $state1 and defined $state2;

    $opt1->update(0, $w1, $g1, $state1);
    $opt2->update(0, $w2, $g2, $state2);
    zip(
        sub {
            my ($s1, $s2) = @_;
            ok(reldiff($s1->aspdl, $s2->aspdl) < 1e-5) if defined $s1 and defined $s2;
        },
        ref $state1 eq 'ARRAY' ? $state1 : [$state1], ref $state2 eq 'ARRAY' ? $state2 : [$state2]
    ) if defined $state1 and defined $state2;
    ok(reldiff($w1->aspdl, $w2->aspdl) < 1e-5);
}

func test_adam()
{
    mx->random->seed(0);
    my $opt1 = 'PerlAdam';
    my $opt2 = 'AI::MXNet::Adam';
    my $shape = [3, 4, 5];
    my @kwargs = ({},
              {'clip_gradient'=> 0.5},
              {'clip_gradient'=> 0.1},
              {'rescale_grad'=> 0.1});
    for my $kwarg (@kwargs)
    {
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(wd => 0.9, %$kwarg), $shape, 'float32');
    }
}

func test_rms()
{
    mx->random->seed(0);
    my $opt1 = 'PerlRMSProp';
    my $opt2 = 'AI::MXNet::RMSProp';
    my $shape = [3, 4, 5];
    my @kwargs = ({},
              {clip_gradient => 0.5},
              {clip_gradient => 0.4, rescale_grad => 0.14},
              {rescale_grad  => 0.8},
              {clip_gradient => 0.5, wd => 0.07},
              {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03},
              {rescale_grad  => 0.8, wd => 0.05},
              {centered => 1},
              {clip_gradient => 0.5, centered => 1},
              {clip_gradient => 0.4, rescale_grad => 0.14, centered => 1},
              {rescale_grad  => 0.8, centered => 1},
              {clip_gradient => 0.5, wd => 0.07, centered => 1},
              {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03, centered => 1},
              {rescale_grad  => 0.8, wd => 0.05, centered => 1},
              {clip_gradient => 0.5, clip_weights => 0.01},
              {clip_gradient => 0.4, rescale_grad => 0.14, clip_weights => 0.01},
              {rescale_grad  => 0.8, clip_weights => 0.01},
              {clip_gradient => 0.5, wd => 0.07, clip_weights => 0.01},
              {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03, clip_weights => 0.01},
              {rescale_grad  => 0.8, wd => 0.05, clip_weights => 0.01},
              {centered => 1, clip_weights => 0.01},
              {clip_gradient => 0.5, centered => 1, clip_weights => 0.01},
              {clip_gradient => 0.4, rescale_grad => 0.14, centered => 1, clip_weights => 0.01},
              {rescale_grad  => 0.8, centered => 1, clip_weights => 0.01},
              {clip_gradient => 0.5, wd => 0.07, centered => 1, clip_weights => 0.01},
              {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03, centered => 1, clip_weights => 0.01},
              {rescale_grad  => 0.8, wd => 0.05, centered => 1, clip_weights => 0.01});
    for my $kwarg (@kwargs)
    {
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(%$kwarg), $shape, 'float32');
    }
}

sub test_sgd
{
    mx->random->seed(0);
    my $opt1 = 'PerlSGD';
    my $opt2 = mx->optimizer->SGD;
    my $shape = [3, 4, 5];
    my @mom_options = ({}, {momentum => 0.9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    my @mp_options = ({}, {multi_precision => 0}, {multi_precision => 1});
    for my $dtype(qw/float16 float32 float64/)
    {
        for my $mom_option (@mom_options)
        {
            for my $cg_option (@cg_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        for my $mp_option (@mp_options)
                        {
                            my %kwarg;
                            %kwarg = (%kwarg, %$mom_option);
                            %kwarg = (%kwarg, %$cg_option);
                            %kwarg = (%kwarg, %$rg_option);
                            %kwarg = (%kwarg, %$wd_option);
                            %kwarg = (%kwarg, %$mp_option);
                            next if (
                                $dtype eq 'float16'
                                    and
                                (not exists $kwarg{multi_precision} or not $kwarg{multi_precision})
                            );
                            compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype);
                        }
                    }
                }
            }
        }
    }
}

sub test_sparse_sgd
{
    mx->random->seed(0);
    my $opt1 = 'PerlSparseSGD';
    my $opt2 = mx->optimizer->SGD;
    my $shape = [3, 4, 5];
    my @mom_options = ({}, {momentum => 0.9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad  => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    my @mp_options = ({}, {multi_precision => 0}, {multi_precision => 1});
    for my $dtype(qw/float32/)
    {
        for my $mom_option (@mom_options)
        {
            for my $cg_option (@cg_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        for my $mp_option (@mp_options)
                        {
                            my %kwarg;
                            %kwarg = (%kwarg, %$mom_option);
                            %kwarg = (%kwarg, %$cg_option);
                            %kwarg = (%kwarg, %$rg_option);
                            %kwarg = (%kwarg, %$wd_option);
                            %kwarg = (%kwarg, %$mp_option);
                            compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype, 'row_sparse', 'row_sparse');
                        }
                    }
                }
            }
        }
    }
}

sub test_std_sparse_sgd
{
    mx->random->seed(0);
    my $opt1 = 'PerlSparseSGD';
    my $opt2 = mx->optimizer->SGD;
    my $shape = [3, 4, 5];
    my @mom_options = ({momentum => 0.9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad  => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    for my $dtype(qw/float32/)
    {
        for my $mom_option (@mom_options)
        {
            for my $cg_option (@cg_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        my %kwarg;
                        %kwarg = (%kwarg, %$mom_option);
                        %kwarg = (%kwarg, %$cg_option);
                        %kwarg = (%kwarg, %$rg_option);
                        %kwarg = (%kwarg, %$wd_option);
                        compare_optimizer($opt1->new(lazy_update => 0, %kwarg), $opt2->new(lazy_update => 0, %kwarg), $shape, $dtype, 'row_sparse', 'row_sparse');
                    }
                }
            }
        }
    }
}

sub test_nag
{
    mx->random->seed(0);
    my $opt1 = 'PerlNAG';
    my $opt2 = mx->optimizer->NAG;
    my $shape = [3, 4, 5];
    my @mom_options = ({}, {momentum => 0.9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    my @mp_options = ({}, {multi_precision => 0}, {multi_precision => 1});
    for my $dtype(qw/float16 float32 float64/)
    {
        for my $mom_option (@mom_options)
        {
            for my $cg_option (@cg_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        for my $mp_option (@mp_options)
                        {
                            my %kwarg;
                            %kwarg = (%kwarg, %$mom_option);
                            %kwarg = (%kwarg, %$cg_option);
                            %kwarg = (%kwarg, %$rg_option);
                            %kwarg = (%kwarg, %$wd_option);
                            # %kwarg = (%kwarg, %$mp_option);
                            next if (
                                $dtype eq 'float16'
                                    and
                                (not exists $kwarg{multi_precision} or not $kwarg{multi_precision})
                            );
                            compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype);
                        }
                    }
                }
            }
        }
    }
}

sub test_ftml
{
    mx->random->seed(0);
    my $opt1 = 'PerlFTML';
    my $opt2 = mx->optimizer->FTML;
    my $shape = [3, 4, 5];
    my @beta1_options = ({}, {beta1 => 0.5}, {beta1 => 0.7});
    my @beta2_options = ({}, {beta1 => 0.8}, {beta1 => 0.9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    for my $dtype(qw/float32/)
    {
        for my $beta1_option (@beta1_options)
        {
            for my $beta2_option (@beta2_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        for my $cg_option (@cg_options)
                        {
                            my %kwarg;
                            %kwarg = (%kwarg, %$beta1_option);
                            %kwarg = (%kwarg, %$beta2_option);
                            %kwarg = (%kwarg, %$cg_option);
                            %kwarg = (%kwarg, %$rg_option);
                            %kwarg = (%kwarg, %$wd_option);
                            compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype);
                        }
                    }
                }
            }
        }
    }
}

sub test_signum
{
    mx->random->seed(0);
    my $opt1 = 'PerlSignum';
    my $opt2 = mx->optimizer->Signum;
    my $shape = [3, 4, 5];
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0.03}, {wd => 0.05}, {wd => 0.07});
    my @wd_lh_options = ({}, {wd_lh => 0.015}, {wd_lh => 0.0});
    my @mom_options = ({}, {momentum => 0.9});
    my @lr_options = ({learning_rate => 0.05}, {learning_rate => 0.01});
    for my $dtype (qw/float32 float64/)
    {
        for my $wd_lh_option (@wd_lh_options)
        {
            for my $mom_option (@mom_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        for my $cg_option (@cg_options)
                        {
                            for my $lr_option (@lr_options)
                            {
                                my %kwarg;
                                %kwarg = (%kwarg, %$wd_lh_option);
                                %kwarg = (%kwarg, %$mom_option);
                                %kwarg = (%kwarg, %$lr_option);
                                %kwarg = (%kwarg, %$cg_option);
                                %kwarg = (%kwarg, %$rg_option);
                                %kwarg = (%kwarg, %$wd_option);
                                compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype);
                            }
                        }
                    }
                }
            }
        }
    }
}


func test_lr_wd_mult()
{
    my $data = mx->sym->Variable('data');
    my $bias = mx->sym->Variable('fc1_bias', lr_mult => 1.0);
    my $fc1  = mx->sym->FullyConnected({ data => $data, bias => $bias, name => 'fc1', num_hidden => 10, lr_mult => 0 });
    my $fc2  = mx->sym->FullyConnected({ data => $fc1, name => 'fc2', num_hidden => 10, wd_mult => 0.5 });

    my $mod = mx->mod->new(symbol => $fc2, label_names => undef);
    $mod->bind(data_shapes => [['data', [5,10]]]);
    $mod->init_params(initializer => mx->init->Uniform(scale => 1.0));
    $mod->init_optimizer(optimizer_params => { learning_rate => "1.0" });
    my %args1 = %{ ($mod->get_params())[0] };
    for my $k (keys %args1)
    {
        $args1{$k} = $args1{$k}->aspdl;
    }
    $mod->forward(AI::MXNet::DataBatch->new(data=>[mx->random->uniform({low=>-1.0, high=>1.0, shape=>[5,10]})], label=>undef), is_train=>1);
    $mod->backward($mod->get_outputs());
    $mod->update();
    my %args2 = %{ ($mod->get_params())[0] };
    for my $k (keys %args2)
    {
        $args2{$k} = $args2{$k}->aspdl;
    }
    is_deeply($mod->_p->_optimizer->lr_mult, { fc1_bias => 1, fc1_weight => 0 }, "lr_mult");
    is_deeply($mod->_p->_optimizer->wd_mult, { fc2_bias => 0.5, fc2_weight => 0.5, fc1_bias => 0, }, "wd_mult");
    ok(almost_equal($args1{fc1_weight}, $args2{fc1_weight}, 1e-10), "fc1_weight");
    ok(!almost_equal($args1{fc1_bias}, $args2{fc1_bias}, 1e-1), "fc1_bias");
    ok(!almost_equal($args1{fc2_weight}, $args2{fc2_weight}, 1e-1), "fc2_weight");
}

sub test_ftrl
{
    mx->random->seed(0);
    my $opt1 = 'PerlFtrl';
    my $opt2 = mx->optimizer->Ftrl;
    my $shape = [3, 4, 5];
    my @kwargs = ({},
              {clip_gradient => 0.5},
              {clip_gradient => 0.4, rescale_grad => 0.14},
              {rescale_grad =>  0.8},
              {clip_gradient =>  0.5, wd => 0.07},
              {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03},
              {rescale_grad => 0.8, wd => 0.05},
              {rescale_grad => 0.8, wd => 0.05, lamda1 => 0.01},
              {clip_gradient => 0.5, wd => 0.07, lamda1 => 1.0});
    for my $kwarg (@kwargs)
    {
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(%$kwarg), $shape, 'float32');
        compare_optimizer($opt1->new(sparse_update=>1, %$kwarg), $opt2->new(%$kwarg), $shape,
                          'float32', 'row_sparse', 'row_sparse');
    }
}

sub test_adagrad
{
    mx->random->seed(0);
    my $opt1 = 'PerlAdaGrad';
    my $opt2 = mx->optimizer->AdaGrad;
    my $shape = [3, 4, 5];
    my @eps_options= ({}, {eps => 1e-9});
    my @cg_options = ({}, {clip_gradient => 0.4}, {clip_gradient => 0.5});
    my @rg_options = ({}, {rescale_grad  => 0.14}, {rescale_grad => 0.8});
    my @wd_options = ({}, {wd => 0});
    for my $dtype(qw/float32/)
    {
        for my $eps_option (@eps_options)
        {
            for my $cg_option (@cg_options)
            {
                for my $rg_option (@rg_options)
                {
                    for my $wd_option (@wd_options)
                    {
                        my %kwarg;
                        %kwarg = (%kwarg, %$eps_option);
                        %kwarg = (%kwarg, %$cg_option);
                        %kwarg = (%kwarg, %$rg_option);
                        %kwarg = (%kwarg, %$wd_option);
                        compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype);
                        if(($wd_option->{wd}//0) == 0)
                        {
                            compare_optimizer($opt1->new(%kwarg), $opt2->new(%kwarg), $shape, $dtype, 'row_sparse', 'row_sparse');
                        }
                    }
                }
            }
        }
    }
}

test_adam();
test_rms();
test_sgd();
test_std_sparse_sgd();
test_sparse_sgd();
test_nag();
test_ftml();
test_signum();
test_ftrl();
test_adagrad();
test_lr_wd_mult();


