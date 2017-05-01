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
        mx->nd->clip($grad, -$self->clip_gradient, $self->clip_gradient, { out => $grad });
    }
    $mean *= $self->beta1;
    $mean += $grad * (1 - $self->beta1);

    $variance *= $self->beta2;
    $variance += (1 - $self->beta2) * mx->nd->square($grad, { out => $grad });

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
        $weight -= $lr * $grad/(mx->nd->sqrt($n) + $self->epsilon);
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
        $delta .= ($self->gamma2) * $delta - $lr * $grad/(mx->nd->sqrt($n - $g*$g) + $self->epsilon);
        $weight += $delta;
    }
    if($self->clip_weights)
    {
        mx->nd->clip($weight, -$self->clip_weights, $self->clip_weights, { out => $weight });
    }
}

package PerlSGD;
# perl reference implemenation of sgd
use Mouse;
extends 'AI::MXNet::Optimizer';
has '+learning_rate' => (default => 0.01);
has 'momentum'       => (is => "ro", isa => "Num",  default => 0);

# Create additional optimizer state: momentum
method create_state(Index $index, AI::MXNet::NDArray $weight)
{
    return undef if $self->momentum == 0;
    return mx->nd->zeros($weight->shape, ctx => $weight->context, dtype => $weight->dtype);
}

method update($index, $weight, $grad, $state)
{
    my $lr = $self->_get_lr($index);
    my $wd = $self->_get_wd($index);
    $self->_update_count($index);
    if($self->momentum == 0)
    {
        if(defined $self->clip_gradient)
        {
            $weight .= ((1 - $lr*$wd)*$weight -
                $lr*mx->nd->clip($grad*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
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
                $lr*mx->nd->clip($grad*$self->rescale_grad, -$self->clip_gradient, $self->clip_gradient)
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

package main;
use Test::More tests => 190;
use AI::MXNet::Base;
use PDL::NiceSlice;
use AI::MXNet::TestUtils qw(same reldiff almost_equal);
use AI::MXNet::Function::Parameters;

func compare_optimizer($opt1, $opt2, $shape)
{
    my $w1 = mx->random->uniform({shape => $shape});
    my $g1 = mx->random->uniform({shape => $shape});

    my $w2 = $w1->copyto(mx->cpu());
    my $g2 = $g1->copyto(mx->cpu());

    my $state1 = $opt1->create_state(0, $w1);
    my $state2 = $opt2->create_state(0, $w2);
    zip(
        sub {
            my ($s1, $s2) = @_;
            ok(same($s1->aspdl, $s2->aspdl))
        },
        ref $state1 eq 'ARRAY' ? $state1 : [$state1], ref $state2 eq 'ARRAY' ? $state2 : [$state2]
    ) if defined $state1 and defined $state2;

    $opt1->update(0, $w1, $g1, $state1);
    $opt2->update(0, $w2, $g2, $state2);
    zip(
        sub {
            my ($s1, $s2) = @_;
            ok(reldiff($s1->aspdl, $s2->aspdl) < 1e-5)
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
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(wd => 0.9, %$kwarg), $shape);
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
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(%$kwarg), $shape);
    }
}


sub test_sgd
{
    mx->random->seed(0);
    my $opt1 = 'PerlSGD';
    my $opt2 = mx->optimizer->SGD;
    my $shape = [3, 4, 5];
    my @kwargs = (
                    {},
                    {momentum => 0.9},
                    {clip_gradient => 0.5},
                    {clip_gradient => 0.4, rescale_grad => 0.14},
                    {rescale_grad  => 0.8},
                    {clip_gradient => 0.5, wd => 0.07},
                    {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03},
                    {rescale_grad  => 0.8, wd => 0.05},
                    {clip_gradient => 0.5, momentum => 0.9},
                    {clip_gradient => 0.4, rescale_grad => 0.14, momentum => 0.9},
                    {rescale_grad  => 0.8, momentum => 0.9},
                    {clip_gradient => 0.5, wd => 0.07, momentum => 0.9},
                    {clip_gradient => 0.4, rescale_grad => 0.14, wd => 0.03, momentum => 0.9},
                    {rescale_grad  => 0.8, wd => 0.05, momentum => 0.9}
    );
    for my $kwarg (@kwargs)
    {
        compare_optimizer($opt1->new(%$kwarg), $opt2->new(%$kwarg), $shape);
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

test_adam();
test_rms();
test_sgd();
test_lr_wd_mult();

