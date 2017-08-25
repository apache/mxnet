use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(almost_equal);
use Test::More tests => 8;

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

test_ndarray_reshape();
test_moveaxis();
test_output();
