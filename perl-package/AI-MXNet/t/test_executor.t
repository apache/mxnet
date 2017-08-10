use strict;
use warnings;
use Test::More tests => 2283;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(reldiff pdl_maximum pdl_minimum);
use PDL;

sub check_bind_with_uniform
{
    my ($uf, $gf, $dim, $sf, $lshape, $rshape) = @_;
    my $shape = (random($dim)*int(1000**(1.0/$dim))+1)->floor->unpdl;
    my $lhs = mx->symbol->Variable('lhs');
    my $rhs = mx->symbol->Variable('rhs');
    my $ret;
    if(defined $sf)
    {
        $ret = &{$sf}($lhs, $rhs);
    }
    else
    {
        $ret = &{$uf}($lhs, $rhs);
    }

    is_deeply($ret->list_arguments(), ['lhs', 'rhs']);
    $lshape //= $shape;
    $rshape //= $shape;

    my $lhs_arr = mx->nd->array(random(reverse (@$lshape)));
    my $rhs_arr = mx->nd->array(random(reverse (@$rshape)));
    my $lhs_grad = mx->nd->empty($lshape);
    my $rhs_grad = mx->nd->empty($rshape);
    my $executor = $ret->bind(
        ctx       => mx->Context('cpu'),
        args      => [$lhs_arr, $rhs_arr],
        args_grad => [$lhs_grad, $rhs_grad]
    );

    my $exec3 = $ret->bind(
        ctx  => mx->Context('cpu'),
        args => [$lhs_arr, $rhs_arr]
    );

    my $exec4 = $ret->bind(
        ctx  => mx->Context('cpu'),
        args => {'rhs' => $rhs_arr, 'lhs' => $lhs_arr},
        args_grad=>{'lhs' => $lhs_grad, 'rhs' => $rhs_grad}
    );

    $executor->forward(1);
    $exec3->forward(1);
    $exec4->forward(1);
    my $out2 = $executor->outputs->[0]->aspdl;
    my $out1 = &{$uf}($lhs_arr->aspdl, $rhs_arr->aspdl);
    my $out3 = $exec3->outputs->[0]->aspdl;
    my $out4 = $exec4->outputs->[0]->aspdl;
    ok(reldiff($out1, $out2) < 1e-6);
    ok(reldiff($out1, $out3) < 1e-6);
    ok(reldiff($out1, $out4) < 1e-6);
    # test gradient

    my $out_grad = mx->nd->ones([reverse @{$out2->shape->unpdl}]);
    my ($lhs_grad2, $rhs_grad2) = &{$gf}(
        $out_grad->aspdl,
        $lhs_arr->aspdl,
        $rhs_arr->aspdl
    );
    $executor->backward([$out_grad]);

    ok(reldiff($lhs_grad->aspdl, $lhs_grad2) < 1e-6);
    ok(reldiff($rhs_grad->aspdl, $rhs_grad2) < 1e-6);
}

sub test_bind
{
    my ($disable_bulk_exec) = @_;
    my ($prev_fwd_var, $prev_bwd_var);
    if($disable_bulk_exec)
    {
        $prev_fwd_var = $ENV{MXNET_EXEC_BULK_FWD_THRESHOLD_TRAIN}//1;
        $prev_bwd_var = $ENV{MXNET_EXEC_BULK_BWD_TRAIN}//1;
        $ENV{MXNET_EXEC_BULK_FWD_THRESHOLD_TRAIN} = 0;
        $ENV{MXNET_EXEC_BULK_BWD_TRAIN} = 0;
    }
    srand(0);
    my $nrepeat = 9;
    my $maxdim = 3;
    for my $repeat (0..$nrepeat)
    {
        for my $dim (1..$maxdim)
        {
            check_bind_with_uniform(sub { my ($x, $y) = @_; $x + $y },
                                    sub { my ($g) = @_; ($g, $g) },
                                    $dim);
            check_bind_with_uniform(sub { my ($x, $y) = @_; $x - $y },
                                    sub { my ($g) = @_; ($g, -$g) },
                                    $dim);
            check_bind_with_uniform(sub { my ($x, $y) = @_; $x * $y },
                                    sub { my ($g, $x, $y) = @_; ($g*$y, $g*$x) },
                                    $dim);
            check_bind_with_uniform(sub { my ($x, $y) = @_; $x / $y },
                                    sub { my ($g, $x, $y) = @_; ($g / $y, -$x * $g/ ($y**2)) },
                                    $dim);
            check_bind_with_uniform(sub { my ($x, $y) = @_; pdl_maximum($x, $y) },
                                    sub { my ($g, $x, $y) = @_; ($g * ($x>$y), $g * ($y>$x)) },
                                    $dim,
                                    sub { $_[0]->maximum($_[1]) });
            check_bind_with_uniform(sub { my ($x, $y) = @_; pdl_minimum($x, $y) },
                                    sub { my ($g, $x, $y) = @_; ($g * ($x<$y), $g * ($y<$x)) },
                                    $dim,
                                    sub { $_[0]->minimum($_[1]) });
        }
    }
    if($disable_bulk_exec)
    {
        $ENV{MXNET_EXEC_BULK_FWD_THRESHOLD_TRAIN} = $prev_fwd_var;
        $ENV{MXNET_EXEC_BULK_BWD_TRAIN}           = $prev_bwd_var;
    }
}


sub test_dot
{
    srand(0);
    my $nrepeat = 9;
    my $maxdim = 4;
    for my $repeat (0..$nrepeat)
    {
        my $shape = (random(3)*500+1)->floor->unpdl;
        check_bind_with_uniform(sub { my ($x, $y) = @_; $x x $y },
                                sub { my ($g, $x, $y) = @_; ($g x $y->transpose, $x->transpose x $g) },
                                2,
                                sub { mx->symbol->dot(@_) },
                                [@{$shape}[0, 1]],
                                [@{$shape}[1, 2]],
        );
    }
    for my $repeat (0..$nrepeat)
    {
        my $shape = (random(1)*500+1)->floor->unpdl;
        check_bind_with_uniform(sub { my ($x, $y) = @_; $x x $y->transpose },
                                sub { my ($g, $x, $y) = @_; ($g * $y, $g * $x) },
                                2,
                                sub { mx->symbol->dot(@_) },
                                [@{$shape}[0]],
                                [@{$shape}[0]],
        );
    }
}

sub test_reshape
{
    my $x = mx->sym->Variable('x');
    my $y = mx->sym->FullyConnected($x, num_hidden=>4);
    my $exe = $y->simple_bind(ctx => mx->cpu(), shapes => { x=>[5,4] }, grad_req=>'null');
    $exe->arg_arrays->[0] .= 1;
    $exe->arg_arrays->[1] .= mx->nd->ones([4,4]);
    $exe->arg_arrays->[2] .= 0;
    my $new_exe = $exe->reshape({ x=>[3,4] });
    $new_exe->forward(0);
    # test sub exec forward
    ok(($new_exe->outputs->[0]->aspdl == 4)->all);
    # test shared memory
    ok(($exe->outputs->[0]->aspdl->slice('X', [0,2]) == 4)->all);
    # test base exec forward
    $exe->forward(0);
    ok(($new_exe->outputs->[0]->aspdl == 4)->all);
}

test_bind(0);
test_bind(1);
test_dot();
test_reshape();
