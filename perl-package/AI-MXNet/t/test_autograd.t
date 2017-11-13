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
use AI::MXNet qw(mx);
use AI::MXNet::AutoGrad qw(autograd);
use AI::MXNet::TestUtils qw(same);
use AI::MXNet::Base;
use Test::More tests => 74;

sub autograd_assert
{
    my $kwargs = {};
    if(ref $_[-1] eq 'HASH') { $kwargs = pop(@_) };
    my @args = @_;
    my $func   = $kwargs->{func};
    my $grad_f = $kwargs->{grad_func};
    my $argnum = $kwargs->{argnum};
    my $grad_func = autograd->grad_and_loss($func, $argnum);
    my ($grad_vals, $output) = $grad_func->(@args);
    my $res = $func->(@args);
    ok(same($output->aspdl, $res->aspdl));
    my $grad_res = $grad_f->(@args);
    ok(@$grad_vals == @$grad_res);
    for(zip($grad_vals, $grad_res)) {
        my ($a, $b) = @$_;
        ok(same($a->aspdl, $b->aspdl));
    }
}

sub test_unary_func
{
    my $check_unary_func = sub {
        my ($x) = @_;
        my $f_exp       = sub { $_[0]->exp };
        my $f_exp_grad  = sub { [$_[0]->exp] };
        autograd_assert($x, { func => $f_exp, grad_func => $f_exp_grad });
        my $f_half      = sub { $_[0]/2 };
        my $f_half_grad = sub { [mx->nd->ones($_[0]->shape) * 0.5] };
        autograd_assert($x, { func => $f_half, grad_func => $f_half_grad });
        my $f_square    = sub { $_[0]**2 };
        my $f_square_grad = sub { [2*$_[0]] };
        autograd_assert($x, { func => $f_square, grad_func => $f_square_grad });
    };
    my $uniform = mx->nd->uniform(shape=>[4, 5]);
    $check_unary_func->($uniform);
    # sparse support
    #my $stypes = ['row_sparse', 'csr', 'default'];
    #for my $stype (@$stypes)
    #{
    #    $check_unary_func->($uniform->tostype($stype));
    #}
}

test_unary_func();

sub test_binary_func
{
    my $check_binary_func = sub {
        my ($x, $y) = @_;
        my $f_add      = sub { $_[0]+$_[1] };
        my $f_add_grad = sub { [map { mx->nd->ones($_->shape) } @_] };
        autograd_assert($x, $y, { func => $f_add, grad_func => $f_add_grad });
        my $f_mul      = sub { $_[0]*$_[1] };
        my $f_mul_grad = sub { [reverse(@_)] };
        autograd_assert($x, $y, { func => $f_mul, grad_func => $f_mul_grad });
        my $f_compose  = sub { $_[0]+$_[0]*$_[1] };
        my $f_compose_grad = sub { [mx->nd->ones($_[0]->shape) + $y, $x] };
        autograd_assert($x, $y, { func => $f_compose, grad_func => $f_compose_grad });
    };
    my $uniform_x = mx->nd->uniform(shape=>[4, 5]);
    my $uniform_y = mx->nd->uniform(shape=>[4, 5]);
    $check_binary_func->($uniform_x, $uniform_y);
    # sparse support
    #my $stypes = ['row_sparse', 'csr', 'default'];
    #for my $stype_x (@$stypes)
    #{
    #    for my $stype_y (@$stypes)
    #    {
    #        my $x = $uniform_x->tostype($stype_x);
    #        my $y = $uniform_y->tostype($stype_y);
    #        $check_binary_func->($x, $y);
    #    }
    #}
}

test_binary_func();

sub test_operator_with_state
{
    my $f_fc = sub {
        my ($a, $b, $weight, $bias) = @_;
        my $x = $a*$b;
        my $fc = mx->nd->FullyConnected(
            $x, $weight, $bias, num_hidden=>32);
        return $fc;
    };

    my $a = mx->nd->uniform(shape=>[64, 50]);
    my $b = mx->nd->uniform(shape=>[64, 50]);
    my $weight = mx->nd->uniform(shape=>[32, 50]);
    my $bias = mx->nd->uniform(shape=>[32]);

    my $grad_func = autograd->grad_and_loss($f_fc);
    my ($grad_vals, $outputs) = $grad_func->($a, $b, $weight, $bias);
}

test_operator_with_state();

sub test_argnum
{
    my $f_with_mode = sub {
        my ($a, $b, $mode) = @_;
        if($mode)
        {
            return $a+$b;
        }
        else
        {
            return $a*$b;
        }
    };
    my $a = mx->nd->uniform(shape=>[3, 2]);
    my $b = mx->nd->uniform(shape=>[3, 2]);
    my $f_add_grad = sub { [map { mx->nd->ones($_->shape) } @_[0,1]] };
    my $f_mul_grad = sub { [reverse(@_[0,1])] };
    autograd_assert($a, $b, 1,
        { argnum=>[0, 1], func=>$f_with_mode, grad_func=>$f_add_grad });
    autograd_assert($a, $b, 0,
        { argnum=>[0, 1], func=>$f_with_mode, grad_func=>$f_mul_grad });
}

test_argnum();

sub test_training
{
    my $x = mx->nd->ones([10, 10]);
    autograd->record(sub {
        my $y = mx->nd->Dropout($x, p=>0.5);
        ok(not ($y->aspdl == $x->aspdl)->all);
        autograd->pause(sub {
            my $y = mx->nd->Dropout($x, p=>0.5);
            ok(($y->aspdl == $x->aspdl)->all);
        });
    });
}

test_training();

sub test_out_grads
{
    my $x = mx->nd->ones([3, 5]);
    my $dx = mx->nd->zeros_like($x);
    autograd->mark_variables([$x], [$dx]);
    my $da;
    my $db = mx->nd->array([1,2,3,4,5]);
    my $dc = mx->nd->array([5,4,3,2,1]);

    autograd->record(sub {
        my ($a, $b, $c) = @{ $x };
        autograd->backward([$a, $b, $c], head_grads => [$da, $db, $dc]);
    });
    ok(($dx->aspdl == pdl(
        [[1,1,1,1,1],
         [1,2,3,4,5],
         [5,4,3,2,1]]))->all);
}

test_out_grads();

sub test_detach_updated_grad
{
    my $x = mx->nd->ones([2, 2]);
    my $dx = mx->nd->zeros_like($x);
    my $y = mx->nd->ones_like($x);
    my $dy = mx->nd->zeros_like($x);
    autograd->mark_variables([$x, $y], [$dx, $dy]);
    ok($x->_fresh_grad == 0);
    ok($y->_fresh_grad == 0);

    autograd->record(sub {
        my $x2 = $x + 2;
        my $y2  = $x2 + $y;
        $y2->backward();
    });
    ok(($dx->aspdl == 1)->all);
    ok($x->_fresh_grad == 1);
    ok($y->_fresh_grad == 1);

    $dx .= 0;
    $x->_fresh_grad(0);
    $y->_fresh_grad(0);
    ok($x->_fresh_grad == 0);
    ok($y->_fresh_grad == 0);

    autograd->record(sub {
        my $x2 = $x + 2;
        $x2 = $x2->detach;
        my $y2  = $x2 + $y;
        $y2->backward();
    });
    ok(($dx->aspdl == 0)->all);
    ok($x->_fresh_grad == 0);
    ok($y->_fresh_grad == 1);
}

test_detach_updated_grad();

sub test_retain_grad
{
    my $x = mx->nd->ones([2, 2]);
    my $dx = mx->nd->zeros([2, 2]);
    autograd->mark_variables([$x], [$dx], grad_reqs=>'add');
    autograd->record(sub {
        my $y = $x + 1;
        $y->backward(retain_graph=>0);
    });
    ok(($dx->aspdl == 1)->all);

    $dx .= 0;
    autograd->record(sub {
        my $y = $x + 1;
        $y->backward(retain_graph=>1);
        $y->backward(retain_graph=>0);
    });
    ok(($dx->aspdl == 2)->all);
    no warnings;
    open(CPERR, ">&STDERR");
    open(STDERR, ">/dev/null");
    eval {
        autograd->record(sub {
            my $y = $x + 1;
            $y->backward();
            $y->backward();
        });
    };
    open(STDERR, ">&CPERR");
    ok($@);
}

test_retain_grad();

sub test_attach_grad
{
    my $check_attach_grad = sub {
        my ($x) = @_;
        ok(not defined $x->grad);
        $x->attach_grad();
        autograd->record(sub {
            my $y = $x * 2;
            ok(not defined $y->grad);
            $y->backward;
        });
        ok(($x->grad->aspdl == 2)->all);
    };
    my $zeros = mx->nd->zeros([10, 10]);
    $check_attach_grad->($zeros);
    # sparse support
    #stypes = ['default', 'row_sparse', 'csr']
    #for stype in stypes:
    #    x = zeros.tostype(stype)
    #    check_attach_grad(x)
}

test_attach_grad();

sub test_is_train
{
    my $x = mx->nd->ones([10, 10]);
    $x->attach_grad();
    autograd->record(sub {
        ok(autograd->is_recording());
        ok(autograd->is_training());
        my $y = mx->nd->Dropout($x, p=>0.5);
        ok($y->aspdl->max == 2 and $y->aspdl->min == 0);
        $y->backward();
        ok(($x->grad->aspdl == $y->aspdl)->all);
        autograd->predict_mode(sub {
            ok(autograd->is_recording());
            ok(not autograd->is_training());
            my $y = mx->nd->Dropout($x, p=>0.5);
            ok(($y->aspdl == $x->aspdl)->all);
            $y->backward(train_mode=>0);
            ok(($x->grad->aspdl == $x->aspdl)->all);
        });
    }, train_mode => 1);

    autograd->record(sub {
        ok(autograd->is_recording());
        ok(not autograd->is_training());
        my $y = mx->nd->Dropout($x, p=>0.5);
        ok(($y->aspdl == $x->aspdl)->all);
        $y->backward(train_mode=>0);
        ok(($x->grad->aspdl == $x->aspdl)->all);

        autograd->train_mode(sub {
            ok(autograd->is_recording);
            ok(autograd->is_training);
            my $y = mx->nd->Dropout($x, p=>0.5);
            ok($y->aspdl->max == 2 and $y->aspdl->min == 0);
            $y->backward;
            ok(($x->grad->aspdl == $y->aspdl)->all);
        });
    }, train_mode => 0);

    ok(not autograd->is_recording);
    ok(not autograd->is_training);
    my $y = mx->nd->Dropout($x, p=>0.5);
    ok(($y->aspdl == $x->aspdl)->all);

    autograd->train_mode(sub {
        ok(not autograd->is_recording);
        ok(autograd->is_training);
        my $y = mx->nd->Dropout($x, p=>0.5);
        ok($y->aspdl->max == 2 and $y->aspdl->min == 0);
    });
}

test_is_train();

sub test_get_symbol
{
    my $x = mx->nd->ones([1]);
    $x->attach_grad;
    my $y;
    autograd->record(sub {
        $y = $x*$x + 2*$x - 1;
    });
    ok(@{ autograd->get_symbol($y)->list_arguments } == 1);

    my $z = mx->nd->ones([1]);
    $z->attach_grad;
    autograd->record(sub {
        $y = $x*$x + 2*$z - 1;
    });
    ok(@{ autograd->get_symbol($y)->list_arguments } == 2);
}

test_get_symbol();

sub test_gradient
{
    my $x = mx->nd->ones([1]);
    $x->attach_grad;
    my $z;
    mx->autograd->record(sub {
        $z = mx->nd->elemwise_add($x->exp, $x);
    });
    my $dx = mx->autograd->grad($z, $x, create_graph=>1);
    ok(abs($dx->asscalar - 3.71828175) < 1e-7);
    $dx->backward;
    ok(abs($x->grad->asscalar - 2.71828175) < 1e-7);
}

test_gradient();
