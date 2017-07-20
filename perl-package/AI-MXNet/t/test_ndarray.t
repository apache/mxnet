use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(almost_equal);
use Test::More tests => 10;

sub test_ndarray_reshape
{
    my $tensor = mx->nd->array([[[1, 2], [3, 4]],
                                [[5, 6], [7, 8]]]);
    my $true_res = mx->nd->arange(stop => 8) + 1;
    is_deeply($tensor->reshape([-1])->aspdl->unpdl, $true_res->aspdl->unpdl);
    $true_res  = mx->nd->array([[1, 2, 3, 4],
                                [5, 6, 7, 8]]);
    is_deeply($tensor->reshape([2, -1])->aspdl->unpdl, $true_res->aspdl->unpdl);
    $true_res  = mx->nd->array([[1, 2],
                                [3, 4],
                                [5, 6],
                                [7, 8]]);
    is_deeply($tensor->reshape([-1, 2])->aspdl->unpdl, $true_res->aspdl->unpdl);
}


sub test_moveaxis
{
    my $X = mx->nd->array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]);
    my $res = $X->moveaxis(0, 2)->aspdl;
    my $true_res = mx->nd->array([[[  1.,   7.],
                                   [  2.,   8.],
                                   [  3.,   9.]],
                                  [[  4.,  10.],
                                   [  5.,  11.],
                                   [  6.,  12.]]]);
    is_deeply($res->unpdl, $true_res->aspdl->unpdl);
    is_deeply($X->moveaxis(2, 0)->shape, [3, 2, 2]);
}


sub test_output
{
    my $shape = [2,2];
    my $ones = mx->nd->ones($shape);
    my $zeros = mx->nd->zeros($shape);
    my $out = mx->nd->zeros($shape);
    mx->nd->ones($shape, out=>$out);
    ok(almost_equal($out->aspdl, $ones->aspdl));
    mx->nd->zeros($shape, out=>$out);
    ok(almost_equal($out->aspdl, $zeros->aspdl));
    mx->nd->full($shape, 2, out=>$out);
    ok(almost_equal($out->aspdl, $ones->aspdl * 2));
}

sub test_cached
{
    my $sym = mx->sym->Convolution(kernel=>[3, 3], num_filter=>10) + 2;
    my $op = mx->nd->CachedOp($sym);
    my $data = mx->nd->ones([3, 4, 10, 10]);
    my $weight = mx->nd->ones([10, 4, 3, 3]);
    my $bias = mx->nd->ones([10]);
    my $o1 = &{$op}($data, $weight, $bias);
    $bias .= 2;
    my $o2 = &{$op}($data, $weight, $bias);
    ok(almost_equal($o2->aspdl, $o1->aspdl+1));
    $o2 .= 0;
    &{$op}($data, $weight, $bias, out=>$o2);
    ok(almost_equal($o2->aspdl, $o1->aspdl+1));
}

test_ndarray_reshape();
test_moveaxis();
test_output();
test_cached();
