use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(same zip);
use Test::More tests => 31;

sub autograd_assert
{
    my ($args, $kwargs) = @_;
    my $func = $kwargs->{func};
    my $grad_f = $kwargs->{grad_func};
    my $argnum = $kwargs->{argnum};

    my $grad_func = mx->contrib->autograd->grad_and_loss($func, $argnum);
    my ($grad_vals, $output) = $grad_func->(@$args);
    my $res = $func->(@$args);
    ok(same($output->aspdl, $res->aspdl));
    my $grad_res = &{$grad_f}(@$args);
    is(scalar(@$grad_vals), scalar(@$grad_res));
    zip(sub {
        ok(same($_[0]->aspdl, $_[1]->aspdl));
    }, $grad_vals, $grad_res);
}

sub test_unary_func
{
    my $x = mx->nd->uniform({ shape=>[4, 5] });
    my $f_exp = sub { $_[0]->exp };
    my $f_exp_grad = sub { [$_[0]->exp] };
    autograd_assert([$x], { func=>$f_exp, grad_func=>$f_exp_grad });
    my $f_half    = sub { $_[0]/2 };
    my $f_half_grad   = sub { [mx->nd->ones($_[0]->shape) * 0.5] };
    autograd_assert([$x], { func=>$f_half, grad_func=>$f_half_grad });
    my $f_square      = sub { $_[0]**2 };
    my $f_square_grad = sub { [2*$_[0]] };
    autograd_assert([$x],{ func=>$f_square, grad_func=>$f_square_grad });
}

test_unary_func();

sub test_binary_func
{
    my $x = mx->nd->uniform({ shape=>[4, 5] });
    my $y = mx->nd->uniform({ shape=>[4, 5] });
    my $f_add      = sub { $_[0] + $_[1] };
    my $f_add_grad = sub { [mx->nd->ones($_[0]->shape), mx->nd->ones($_[1]->shape)] };
    autograd_assert([$x, $y], { func=>$f_add, grad_func=>$f_add_grad });
    my $f_mul      = sub { $_[0] * $_[1] };
    my $f_mul_grad = sub { [$_[1], $_[0]] };
    autograd_assert([$x, $y], { func=>$f_mul, grad_func=>$f_mul_grad });
    my $f_compose  = sub { $_[0] + $_[0]*$_[1] };
    my $f_compose_grad = sub { [mx->nd->ones($_[0]->shape) + $_[1], $_[0]] };
    autograd_assert([$x, $y], { func=>$f_compose, grad_func=>$f_compose_grad });
}

test_binary_func();

sub test_argnum
{

    my $f_with_mode = sub { my ($a, $b, $mode) = @_;
        if($mode)
        {
            return $a+$b;
        }
        else
        {
            return $a*$b
        }
    };

    my $a = mx->nd->uniform({ shape=>[3, 2] });
    my $b = mx->nd->uniform({ shape=>[3, 2] });
    my $f_add_grad = sub { [mx->nd->ones($_[0]->shape), mx->nd->ones($_[1]->shape)] };
    my $f_mul_grad = sub { [$_[1], $_[0]] };
    autograd_assert([$a, $b, 1],
        { argnum=>[0, 1], func=>$f_with_mode, grad_func=>$f_add_grad });
    autograd_assert([$a, $b, 0],
        { argnum=>[0, 1], func=>$f_with_mode, grad_func=>$f_mul_grad });
}

test_argnum();

sub test_training
{
    my $x = mx->nd->ones([10, 10]);
    mx->contrib->autograd->set_is_training(1);
    my $y = mx->nd->Dropout($x, { p=>0.5 });
    ok(not ($y->aspdl== $x->aspdl)->all);
    mx->contrib->autograd->set_is_training(0);
    $y = mx->nd->Dropout($x, { p=>0.5 });
    ok(($y->aspdl== $x->aspdl)->all);
}

test_training();

