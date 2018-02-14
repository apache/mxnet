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

package AI::MXNet::Optimizer;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray;
use AI::MXNet::Random;
use List::Util qw(max);

=head1 NAME

    AI::MXNet::Optimizer - Common Optimization algorithms with regularizations.

=head1  DESCRIPTION

    Common Optimization algorithms with regularizations.
=cut

use Mouse;
use AI::MXNet::Function::Parameters;
my %opt_registry;
method get_opt_registry()
{
    return \%opt_registry;
}

method register()
{
    my $name = $self;
    ($name) = $name =~ /::(\w+)$/;
    {  no strict 'refs'; *{__PACKAGE__."::$name"} = sub { $self }; }
    $name = lc $name;
    if(exists $opt_registry{ $name })
    {
        my $existing = $opt_registry{ $name };
        warn(
            "WARNING: New optimizer $self.$name"
            ."is overriding existing optimizer $existing.$name"
        );
    }
    $opt_registry{ $name } = $self;
}

=head2 create_optimizer

        Create an optimizer with specified name.

        Parameters
        ----------
        name: str
            Name of required optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        rescale_grad : float
            Rescaling factor on gradient. Normally should be 1/batch_size.

        kwargs: dict
            Parameters for optimizer

        Returns
        -------
        opt : Optimizer
            The result optimizer.
=cut

method create_optimizer(Str $name, %kwargs)
{
    if(exists $opt_registry{ lc $name })
    {
        my $rescale_grad = delete($kwargs{rescale_grad})//1;
        return $opt_registry{ lc $name }->new(
            rescale_grad => $rescale_grad,
            %kwargs
        );
    }
    confess("Cannot find optimizer $name");
}

*create = \&create_optimizer;

has 'rescale_grad'        => (is => "rw", isa => "Num", default=>1);
has 'lr'                  => (is => "rw", isa => "Num");
has 'learning_rate'       => (is => "rw", isa => "Num", default => 0.01);
has 'lr_scheduler'        => (is => "rw", isa => "Maybe[AI::MXNet::LRScheduler]");
has 'wd'                  => (is => "rw", isa => "Num", default => 0);
has 'lr_mult'             => (is => "rw", isa => "HashRef", default => sub { +{} });
has 'wd_mult'             => (is => "rw", isa => "HashRef", , default => sub { +{} });
has 'num_update'          => (is => "rw", isa => "Int");
has 'begin_num_update'    => (is => "rw", isa => "Int", default => 0);
has '_index_update_count' => (is => "rw", isa => "HashRef", default => sub { +{} });
has 'clip_gradient'       => (is => "rw", isa => "Maybe[Num]");
has 'param_idx2name'      => (is => "rw", isa => "HashRef[Str]", default => sub { +{} });
has 'idx2name'            => (is => "rw", isa => "HashRef[Str]");
has 'sym'                 => (is => "rw", isa => "Maybe[AI::MXNet::Symbol]");
has 'param_dict'          => (is => "rw", isa => "HashRef", default => sub { +{} });

sub BUILD
{
    my $self = shift;
    if($self->lr_scheduler)
    {
        $self->lr_scheduler->base_lr($self->learning_rate);
    }
    $self->lr($self->learning_rate);
    $self->num_update($self->begin_num_update);
    $self->idx2name({ %{ $self->param_idx2name } });
    $self->set_lr_mult({});
    $self->set_wd_mult({});
}
# Create additional optimizer state such as momentum.
# override in implementations.
method create_state($index, $weight){}

# Update the parameters. override in implementations
method update($index, $weight, $grad, $state){}

# set lr scale is deprecated. Use set_lr_mult instead.
method set_lr_scale($args_lrscale)
{
    Carp::cluck("set lr scale is deprecated. Use set_lr_mult instead.");
}

=head2 set_lr_mult

        Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
=cut

method set_lr_mult(HashRef[Num] $args_lr_mult)
{
    $self->lr_mult({});
    if($self->sym)
    {
        my $attr = $self->sym->attr_dict();
        for my $name (@{ $self->sym->list_arguments() })
        {
            if(exists $attr->{ $name } and exists $attr->{ $name }{ __lr_mult__ })
            {
                $self->lr_mult->{ $name } = $attr->{ $name }{ __lr_mult__ };
            }
        }
    }
    $self->lr_mult({ %{ $self->lr_mult }, %{ $args_lr_mult } });
}

=head2 set_wd_mult

        Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
=cut

method set_wd_mult(HashRef[Num] $args_wd_mult)
{
    $self->wd_mult({});
    for my $n (values %{ $self->idx2name })
    {
        if(not $n =~ /(?:_weight|_gamma)$/)
        {
            $self->wd_mult->{ $n } = 0;
        }
    }
    if($self->sym)
    {
        my $attr = $self->sym->attr_dict();
        for my $name (@{ $self->sym->list_arguments() })
        {
            if(exists $attr->{ $name } and exists $attr->{ $name }{ __wd_mult__ })
            {
                $self->wd_mult->{ $name } = $attr->{ $name }{ __wd_mult__ };
            }
        }
    }
    $self->wd_mult({ %{ $self->wd_mult }, %{ $args_wd_mult } });
}

method _update_count(Index $index)
{
    if(not exists $self->_index_update_count->{ $index })
    {
        $self->_index_update_count->{ $index } = $self->begin_num_update;
    }
    $self->_index_update_count->{ $index } += 1;
    $self->num_update(max($self->_index_update_count->{ $index }, $self->num_update));
}

method _get_lr(Index $index)
{
    my $lr;
    if($self->lr_scheduler)
    {
        $lr = $self->lr_scheduler->($self->num_update);
    }
    else
    {
        $lr = $self->lr;
    }

    if(exists $self->param_dict->{ $index })
    {
        $lr *= $self->param_dict->{ $index }->lr_mult;
    }
    elsif(exists $self->lr_mult->{ $index })
    {
        $lr *= $self->lr_mult->{ $index };
    }
    elsif(exists $self->idx2name->{ $index })
    {
        $lr *= $self->lr_mult->{ $self->idx2name->{ $index } }//1;
    }
    return $lr;
}

method _get_wd(Index $index)
{
    my $wd = $self->wd;
    if(exists $self->param_dict->{ $index })
    {
        $wd *= $self->param_dict->{ $index }->wd_mult;
    }
    elsif(exists $self->wd_mult->{ $index })
    {
        $wd *= $self->wd_mult->{ $index };
    }
    elsif(exists $self->idx2name->{ $index })
    {
        $wd *= $self->wd_mult->{ $self->idx2name->{ $index } }//1;
    }
    return $wd;
}

=head1 NAME

    AI::MXNet::SGD - A very simple SGD optimizer with momentum and weight regularization.
=cut

=head1 DESCRIPTION

    A very simple SGD optimizer with momentum and weight regularization.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : hash of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta

    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.
        False results in using the same precision as the weights (default),
        True makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
=cut

package AI::MXNet::SGD;
use Mouse;
extends 'AI::MXNet::Optimizer';

has 'kwargs'   => (is => "rw", isa => "HashRef[Num]");
has 'momentum' => (is => "rw", isa => "Num", default => 0);
has 'multi_precision' => (is => "ro", isa => "Bool", default => 0);

sub BUILD
{
    my $self = shift;
    $self->kwargs({});
    if($self->momentum)
    {
        $self->kwargs->{momentum} = $self->momentum;
    }
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
}

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    my $momentum;
    my $weight_master_copy;
    if($self->multi_precision and $weight->dtype eq 'float16')
    {
        my $weight_master_copy = AI::MXNet::NDArray->array($weight, ctx => $weight->context, dtype => 'float32');
        if($self->momentum != 0)
        {
            $momentum = AI::MXNet::NDArray->zeros($weight->shape, ctx => $weight->context, dtype => 'float32');
        }
        return [$momentum, $weight_master_copy];
    }
    if($weight->dtype eq 'float16' and not $self->multi_precision)
    {
        AI::MXNet::Logging->warning(
            "Accumulating with float16 in optimizer can lead to ".
            "poor accuracy or slow convergence. ".
            "Consider using multi_precision=True option of the ".
            "SGD optimizer"
        );
    }
    if($self->momentum != 0)
    {
        $momentum = AI::MXNet::NDArray->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype);
    }
    return $momentum;
}

method update(
    Index                     $index,
    AI::MXNet::NDArray        $weight,
    AI::MXNet::NDArray        $grad,
    Maybe[AI::MXNet::NDArray|ArrayRef[Maybe[AI::MXNet::NDArray]]] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $kwargs = {
        out => $weight,
        lr  => $lr,
        wd  => $wd,
        rescale_grad => $self->rescale_grad,
        %{ $self->kwargs }
    };
    my $use_multi_precision = ref($state) eq 'ARRAY';
    if(not $use_multi_precision)
    {
        if(defined $state)
        {
            AI::MXNet::NDArray->sgd_mom_update(
                $weight, $grad, $state, $kwargs
            );
        }
        else
        {
            AI::MXNet::NDArray->sgd_update(
                $weight, $grad, $kwargs
            );
        }
    }
    else
    {
        if(defined $state->[0])
        {
            AI::MXNet::NDArray->mp_sgd_mom_update(
                $weight, $grad, $state->[0], $state->[1], $kwargs
            );
        }
        else
        {
            AI::MXNet::NDArray->mp_sgd_update(
                $weight, $grad, $state->[1], $kwargs
            );
        }
    }
}

__PACKAGE__->register;

package AI::MXNet::DCASGD;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Optimizer';

=head1 NAME

    AI::MXNet::DCASGD - DCASGD optimizer with momentum and weight regularization.
=cut

=head1 DESCRIPTION

    DCASGD optimizer with momentum and weight regularization.

    Implements paper "Asynchronous Stochastic Gradient Descent with
                    Delay Compensation for Distributed Deep Learning"

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    lamda : float, optional
       scale DC value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : hash ref of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
=cut
has 'momentum'        => (is => 'ro', isa => 'Num', default => 0);
has 'lamda'           => (is => 'ro', isa => 'Num', default => 0.04);
has 'weight_previous' => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->weight_previous({});
}

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
        return [
            $self->momentum ? AI::MXNet::NDArray->zeros(
                $weight->shape, ctx => $weight->context, dtype => $weight->dtype
            ) : undef,
            $weight->copy
        ];
}

method update(
    Index                     $index,
    AI::MXNet::NDArray        $weight,
    AI::MXNet::NDArray        $grad,
    Maybe[AI::MXNet::NDArray|ArrayRef[Maybe[AI::MXNet::NDArray]]] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
            $self->clip_gradient
        );
    }
    my ($mom, $weight_previous) = @{ $state };
    if(defined $mom)
    {
        $mom *= $self->momentum;
        $mom += -$lr * (
                $grad + $wd * $weight
                    +
                $self->lamda * $grad * $grad * ($weight - $weight_previous)
        );
    }
    else
    {
        assert($self->momentum == 0);
        $mom = -$lr * (
                $grad + $wd * $weight
                    +
                $self->lamda * $grad * $grad * ($weight - $weight_previous)
        );
    }
    $weight_previous .= $weight;
    $weight += $mom;
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::NAG - SGD with Nesterov weight handling.
=cut

=head1 DESCRIPTION

    It is implemented according to
    https://github.com/torch/optim/blob/master/sgd.lua
=cut

package AI::MXNet::NAG;
use Mouse;

extends 'AI::MXNet::SGD';

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray|Undef $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad = $grad * $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
            $self->clip_gradient
        );
    }
    if($state)
    {
        my $mom  = $state;
        $mom    *= $self->momentum;
        $grad   += $wd * $weight;
        $mom    += $grad;
        $grad   += $self->momentum * $mom;
        $weight += -$lr * $grad;
    }
    else
    {
        confess("momentum != 0") unless $self->momentum == 0;
        $weight += -$lr * ($grad + $wd * $weight);
    }
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::SLGD - Stochastic Langevin Dynamics Updater to sample from a distribution.
=cut

=head1 DESCRIPTION

    Stochastic Langevin Dynamics Updater to sample from a distribution.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
=cut

package AI::MXNet::SLGD;
use Mouse;

extends 'AI::MXNet::Optimizer';

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return undef;
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray|Undef $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    $weight +=  - $lr/2 * ($grad + $wd * $weight)
                    +
                AI::MXNet::Random->normal(
                        0, sqrt($lr),
                        $weight->shape,
                        $weight->context
                );
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::Adam - Adam optimizer as described in [King2014]_.
=cut

=head1 DESCRIPTION

    Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    the code in this class was adapted from
    https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L765

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.001.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.999.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::Adam;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'kwargs'   => (is => "rw", isa => "HashRef[Num]");
has '+learning_rate' => (default => 0.001);
has 'beta1'    => (is => "rw", isa => "Num", default => 0.9);
has 'beta2'    => (is => "rw", isa => "Num", default => 0.999);
has 'epsilon'  => (is => "rw", isa => "Num", default => 1e-8);
has 'decay_factor'  => (is => "rw", isa => "Num", default => (1 - 1e-8));

sub BUILD
{
    my $self = shift;
    $self->kwargs({
        beta1   => $self->beta1,
        beta2   => $self->beta2,
        epsilon => $self->epsilon
    });
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
}

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            ),  # mean
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            )  # variance
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my $t = $self->_index_update_count->{$index};
    my $coef1 = 1 - $self->beta1**$t;
    my $coef2 = 1 - $self->beta2**$t;
    $lr *= sqrt($coef2)/$coef1;
    my ($mean, $var) = @{ $state };
    AI::MXNet::NDArray->adam_update(
        $weight, $grad, $mean, $var,
        {
            out => $weight,
            lr  => $lr,
            wd  => $wd,
            rescale_grad => $self->rescale_grad,
            %{ $self->kwargs }
        }
    );
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::AdaGrad - AdaGrad optimizer of Duchi et al., 2011
=cut

=head1 DESCRIPTION

    AdaGrad optimizer of Duchi et al., 2011,

    This code follows the version in http://arxiv.org/pdf/1212.5701v1.pdf  Eq(5)
    by Matthew D. Zeiler, 2012. AdaGrad will help the network to converge faster
    in some cases.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.05.

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    eps: float, optional
        A small float number to make the updating processing stable
        Default value is set to 1e-7.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::AdaGrad;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'float_stable_eps'    => (is => "rw", isa => "Num", default => 1e-7);
has '+learning_rate'       => (default => 0.05);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
    );  # history
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    my $history = $state;
    $history += ($grad * $grad);
    $weight  += -$lr
                    *
                (
                    $grad
                        /
                    AI::MXNet::NDArray->sqrt(
                        $history
                            +
                        $self->float_stable_eps
                    )
                        +
                    $wd * $weight
                );
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::RMSProp - RMSProp optimizer of Tieleman & Hinton, 2012.
=cut

=head1 DESCRIPTION

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
        decay factor of moving average for gradient^2.
        Default value is set to 0.9.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
        Only used if centered=True
    epsilon : float, optional
        Default value is set to 1e-8.
    centered : bool, optional
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

package AI::MXNet::RMSProp;
use Mouse;

extends 'AI::MXNet::Optimizer';

has '+learning_rate' => (default => 0.001);
has 'gamma1'         => (is => "ro", isa => "Num",  default => 0.9);
has 'gamma2'         => (is => "ro", isa => "Num",  default => 0.9);
has 'epsilon'        => (is => "ro", isa => "Num",  default => 1e-8);
has 'centered'       => (is => "ro", isa => "Bool", default => 0);
has 'clip_weights'   => (is => "ro", isa => "Num");
has 'kwargs'         => (is => "rw", init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->kwargs({
        gamma1       => $self->gamma1,
        epsilon      => $self->epsilon
    });
    if($self->centered)
    {
        $self->kwargs->{gamma2} = $self->gamma2;
    }
    if($self->clip_gradient)
    {
        $self->kwargs->{clip_gradient} = $self->clip_gradient;
    }
    if($self->clip_weights)
    {
        $self->kwargs->{clip_weights} = $self->clip_weights;
    }
}

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

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    my ($n, $g, $delta) = @{ $state };
    if($self->centered)
    {
        AI::MXNet::NDArray->rmspropalex_update(
            $weight, $grad, $n, $g, $delta,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                rescale_grad => $self->rescale_grad,
                %{ $self->kwargs }
            }
        );
    }
    else
    {
        AI::MXNet::NDArray->rmsprop_update(
            $weight, $grad, $n,
            {
                out => $weight,
                lr  => $lr,
                wd  => $wd,
                rescale_grad => $self->rescale_grad,
                %{ $self->kwargs }
            }
        );
    }
}

__PACKAGE__->register;

=head1 NAME

    AI::MXNet::AdaDelta - AdaDelta optimizer.
=cut

=head1 DESCRIPTION

    AdaDelta optimizer as described in
    Zeiler, M. D. (2012).
    *ADADELTA: An adaptive learning rate method.*

    http://arxiv.org/abs/1212.5701

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta x
    epsilon : float
        The constant as described in the thesis
    wd : float
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
=cut
package AI::MXNet::AdaDelta;
use Mouse;

extends 'AI::MXNet::Optimizer';

has 'rho'    => (is => "rw", isa => "Num", default => 0.9);
has 'epsilon'    => (is => "rw", isa => "Num", default => 1e-5);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            ),  # accumulated g
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            )   # accumulated delta
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    my ($acc_g, $acc_delta) = @{ $state };
    $acc_g .= $self->rho * $acc_g + (1 - $self->rho) * $grad * $grad;
    my $current_delta = ($acc_delta + $self->epsilon)->sqrt
                            /
                        ($acc_g + $self->epsilon)->sqrt
                            *
                        $grad;
    $acc_delta .= $self->rho * $acc_delta + (1 - $self->rho) * $current_delta * $current_delta;
    $weight -= $current_delta + $wd * $weight;
}

__PACKAGE__->register;

# For test use
package AI::MXNet::Test;
use Mouse;

extends 'AI::MXNet::Optimizer';

# Create a state to duplicate weight
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
    );
}

# performs w += rescale_grad * grad
method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    AI::MXNet::NDArray $state
)
{
    $weight += $grad * $self->rescale_grad;
    $state .= $weight;
}

__PACKAGE__->register;

package AI::MXNet::Ftrl;

=head1 NAME

    AI::MXNet::Ftrl
=cut

=head1 DESCRIPTION

    Reference:Ad Click Prediction: a View from the Trenches

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.

    learning_rate : float, optional
        The initial learning rate.

    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    eta_{t,i}=frac{learning_rate}{beta+sqrt{sum_{s=1^}tg_{s,i}^t}
=cut

use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.1);
has 'beta'           => (is => "ro", isa => "Num",  default => 1);
has 'lambda1'        => (is => "ro", isa => "Num",  default => 0.9);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            ),  # dn
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context
            )   # n
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    $self->_update_count($index);
    my $wd = $self->_get_wd($index);
    my $lr = $self->_get_lr($index);
    $grad *= $self->rescale_grad;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    my ($dn, $n) = @{ $state };
    $dn += $grad - (($n + $grad * $grad)->sqrt - $n->sqrt) * $weight / $lr;
    $n += $grad * $grad;

    $weight .= ($dn->sign * $self->lamda1 - $dn)
                    /
               (($self->beta + $n->sqrt) / $lr + $wd) * ($dn->abs > $self->lamda1);
}

__PACKAGE__->register;

package AI::MXNet::Adamax;

=head1 NAME

    AI::MXNet::Adamax
=cut

=head1 DESCRIPTION

    It is a variant of Adam based on the infinity norm
    available at http://arxiv.org/abs/1412.6980 Section 7.

    This optimizer accepts the following parameters in addition to those accepted
    AI::MXNet::Optimizer.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
=cut

use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.002);
has 'beta1'          => (is => "ro", isa => "Num",  default => 0.9);
has 'beta2'          => (is => "ro", isa => "Num",  default => 0.999);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            ),  # mean
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            )   # variance
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $wd = $self->_get_wd($index);
    my $lr = $self->_get_lr($index);
    $self->_update_count($index);
    my $t = $self->_index_update_count->{$index};
    $lr /= (1 - $self->beta1**$t);

    $grad = $grad * $self->rescale_grad + $wd * $weight;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }

    # update m_t and u_t
    my($m_t, $u_t) = @{ $state };
    $m_t .= $self->beta1 * $m_t + (1 - $self->beta1) * $grad;
    $u_t .= AI::MXNet::NDArray->maximum($self->beta2 * $u_t, $grad->abs);

    # update weight
    $weight -= $lr * $m_t / $u_t;
}

__PACKAGE__->register;

package AI::MXNet::Nadam;

=head1 NAME

    AI::MXNet::Nadam
=cut

=head1 DESCRIPTION

    The Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum available
    at http://cs229.stanford.edu/proj2015/054_report.pdf.

    This optimizer accepts the following parameters in addition to those accepted
    AI::MXNet::Optimizer.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    schedule_decay : float, optional
        Exponential decay rate for the momentum schedule
=cut

use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.001);
has 'beta1'          => (is => "ro", isa => "Num",  default => 0.9);
has 'beta2'          => (is => "ro", isa => "Num",  default => 0.999);
has 'epsilon'        => (is => "ro", isa => "Num",  default => 1e-8);
has 'schedule_decay' => (is => "ro", isa => "Num",  default => 0.004);
has 'm_schedule'     => (is => "rw", default => 1, init_arg => undef);

method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return [
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            ),  # mean
            AI::MXNet::NDArray->zeros(
                $weight->shape,
                ctx => $weight->context,
                dtype => $weight->dtype
            )   # variance
    ];
}

method update(
    Index $index,
    AI::MXNet::NDArray $weight,
    AI::MXNet::NDArray $grad,
    ArrayRef[AI::MXNet::NDArray] $state
)
{
    my $wd = $self->_get_wd($index);
    my $lr = $self->_get_lr($index);
    $self->_update_count($index);
    my $t = $self->_index_update_count->{$index};
    $grad = $grad * $self->rescale_grad + $wd * $weight;
    if($self->clip_gradient)
    {
        $grad = AI::MXNet::NDArray->clip(
            $grad,
            -$self->clip_gradient,
             $self->clip_gradient
        );
    }
    # warming momentum schedule
    my $momentum_t    = $self->beta1 * (1 - 0.5 * (0.96**($t * $self->schedule_decay)));
    my $momentum_t_1  = $self->beta1 * (1 - 0.5 * (0.96**(($t + 1) * $self->schedule_decay)));
    $self->m_schedule = $self->m_schedule * $momentum_t;
    my $m_schedule_next  = $self->m_schedule * $momentum_t_1;

    # update m_t and v_t
    my ($m_t, $v_t) = @{ $state };
    $m_t .= $self->beta1 * $m_t + (1 - $self->beta1) * $grad;
    $v_t .= $self->beta2 * $v_t + (1 - $self->beta2) * $grad * $grad;

    my $grad_prime = $grad / (1 - $self->m_schedule);
    my $m_t_prime  = $m_t  / (1 - $m_schedule_next);
    my $v_t_prime  = $v_t  / (1 - $self->beta2**$t);
    my $m_t_bar    = (1 - $momentum_t) * $grad_prime + $momentum_t_1 * $m_t_prime;

    # update weight
    $weight -= $lr * $m_t_bar / (sqrt($v_t_prime) + $self->epsilon);
}

__PACKAGE__->register;

# updater for kvstore
package AI::MXNet::Updater;
use Mouse;
use Storable qw(thaw freeze);
use overload "&{}" => sub { my $self = shift; sub { $self->call(@_) } },
             fallback => 1;

has "optimizer"     => (is => "rw", isa => "AI::MXNet::Optimizer");
has "states"        => (is => "rw", isa => "HashRef", default => sub { +{} });
has "states_synced" => (is => "rw", isa => "HashRef", default => sub { +{} });

method call(Index $index, AI::MXNet::NDArray $grad, AI::MXNet::NDArray $weight)
{
    if(not exists $self->states->{ $index })
    {
        $self->states->{ $index } = $self->optimizer->create_state($index, $weight);
        $self->states_synced->{ $index } = 1;
    }
    elsif(not $self->states_synced->{ $index })
    {
        $self->states->{ $index } = $self->sync_state_context($self->states->{ $index }, $weight->context);
        $self->states_synced->{ $index } = 1;
    }
    $self->optimizer->update($index, $weight, $grad, $self->states->{ $index });
}
*slice = *call;

method sync_state_context(Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]] $state, AI::MXNet::Context $context)
{
    if(blessed $state)
    {
        return $state->as_in_context($context);
    }
    elsif(ref $state)
    {
        return [map { $self->sync_state_context($_, $context) } @{ $state }];
    }
    return $state;
}

=head2 set_states

    Sets updater states.
=cut

method set_states($states)
{
    my $thawed_states = thaw($states);
    my ($optimizer);
    if(ref $thawed_states eq 'ARRAY')
    {
        ($thawed_states, $optimizer) = @{ $thawed_states };
        $self->optimizer($optimizer);
    }
    $self->states($thawed_states);
    %{ $self->states_synced } = map { $_ => 0 } keys %{ $thawed_states };
}

=head2 get_states

        Gets updater states.

        Parameters
        ----------
        dump_optimizer : bool, default False
            Whether to also save the optimizer itself. This would also save optimizer
            information such as learning rate and weight decay schedules.
=cut

method get_states(Bool $dump_optimizer=0)
{
    return freeze($dump_optimizer ? [$self->states, $self->optimizer] : $self->states);
}

package AI::MXNet::Optimizer;

method get_updater(AI::MXNet::Optimizer $optimizer)
{
    return AI::MXNet::Updater->new(optimizer => $optimizer);
}

1;
