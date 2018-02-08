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
use AI::MXNet::TestUtils qw(almost_equal same);
use Test::More tests => 17;

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
    my $o1 = $op->($data, $weight, $bias);
    $bias .= 2;
    my $o2 = $op->($data, $weight, $bias);
    ok(almost_equal($o2->aspdl, $o1->aspdl+1));
    $o2 .= 0;
    $op->($data, $weight, $bias, out=>$o2);
    ok(almost_equal($o2->aspdl, $o1->aspdl+1));

    $weight->attach_grad();
    $bias->attach_grad();
    my $o;
    mx->autograd->record(sub {
        $bias = $bias + 1;
        $o = $op->($data, $weight, $bias);
        $o = $o * 2;
        $o->backward();
    });

    mx->autograd->record(sub {
        $bias = $bias + 1;
        $o = $op->($data, $weight, $bias);
        $o = $o * 2;
        $o->backward(retain_graph=>1);
        $o->backward();
    });

    # try a different shape
    $data = mx->nd->ones([5, 2, 10, 10]);
    $weight = mx->nd->ones([10, 2, 3, 3]);
    $bias = mx->nd->ones([10]);
    $data->attach_grad;

    mx->autograd->record(sub {
        $bias = $bias + 1;
        $o = $op->($data, $weight, $bias);
        $o = $o * 2;
        $o->backward();
    });
}

sub test_ndarray_slice
{
    my $shape = [10];
    my $A = mx->random->uniform(-10, 10, $shape);
    my $A2 = $A->aspdl;
    ok(same($A->slice([3,7])->aspdl, $A2->slice([3, 7])));
    $A2->slice([3, 7]) *= 10;
    $A->slice([3,7]) .= $A2->slice([3, 7]);
    ok(same($A->slice([3,7])->aspdl, $A2->slice([3, 7])));

    $shape = [3,4,5,6,7];
    $A = mx->nd->random->uniform(shape=>$shape);
    $A2 = $A->aspdl;

    ok(same($A->slice([1], [3,3], 'X', [1,4], 'X')->aspdl, $A2->slice('X', [1,4], 'X', [3,3], [1])));
    ok(($A->slice([1], [3,3], 'X', [1,4], 'X') == mx->nd->array($A2->slice('X', [1,4], 'X', [3,3], [1])))->aspdl->all);

    ok($A->slice(1,2,3,4,5)->asscalar() == $A2->at(5, 4, 3, 2, 1));

    my $a = mx->nd->array([[0, 1], [2, 3]]);
    ok(($a->slice([[1, 1, 0], [0, 1, 0]])->aspdl == mx->nd->array([2, 3, 0])->aspdl)->all);
    ok(($a->slice([mx->nd->array([1, 1, 0]), mx->nd->array([0, 1, 0])])->aspdl == mx->nd->array([2, 3, 0])->aspdl)->all);
}

test_ndarray_slice();
test_ndarray_reshape();
test_moveaxis();
test_output();
test_cached();
