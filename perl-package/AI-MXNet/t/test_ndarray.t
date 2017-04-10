use AI::MXNet qw(mx);
use Test::More tests => 5;

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

test_ndarray_reshape();
test_moveaxis();