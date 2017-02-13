package PerlAdam;
use strict;
use warnings;
use AI::MXNet qw(mx);
use Mouse;
use Method::Signatures;
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
    $grad *= $self->rescale_grad;
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

    my $wd = $self->_get_wd($index);
    if($wd > 0)
    {
        $weight -= ($lr * $wd) * $weight;
    }
}

package main;
use Test::More tests => 25;
use AI::MXNet::Base;
use Data::Dumper;
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
        $state1, $state2
    );

    $opt1->update(0, $w1, $g1, $state1);
    $opt2->update(0, $w2, $g2, $state2);
    zip(
        sub {
            my ($s1, $s2) = @_;
            ok(reldiff($s1->aspdl, $s2->aspdl) < 1e-5)
        },
        $state1, $state2
    );
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
test_lr_wd_mult();

