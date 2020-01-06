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
use Scalar::Util qw(blessed);
use Test::More 'no_plan';
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(zip assert enumerate same rand_shape_2d rand_shape_3d
    rand_sparse_ndarray random_arrays almost_equal rand_ndarray randint allclose dies_ok);
use AI::MXNet::Base qw(pones pzeros pdl product rand_sparse);
$ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE} = 0;
$ENV{MXNET_SUBGRAPH_VERBOSE} = 0;


sub sparse_nd_ones
{
    my ($shape, $stype) = @_;
    return mx->nd->ones($shape)->tostype($stype);
}

sub test_sparse_nd_elemwise_add
{
    my $check_sparse_nd_elemwise_binary = sub {
        my ($shapes, $stypes, $f, $g) = @_;
        # generate inputs
        my @nds;
        enumerate(sub {
            my ($i, $stype) = @_;
            my $nd;
            if($stype eq 'row_sparse')
            {
                ($nd) = rand_sparse_ndarray($shapes->[$i], $stype);
            }
            elsif($stype eq 'default')
            {
                $nd = mx->nd->array(random_arrays($shapes->[$i]), dtype => 'float32');
            }
            else
            {
                die;
            }
            push @nds, $nd;
        }, $stypes);
        # check result
        my $test = $f->($nds[0], $nds[1]);
        ok(almost_equal($test->aspdl, $g->($nds[0]->aspdl, $nds[1]->aspdl)));
    };
    my $num_repeats = 2;
    my $g = sub { $_[0] + $_[1] };
    my $op = sub { mx->nd->elemwise_add(@_) };
    for my $i (0..$num_repeats)
    {
        my $shape = rand_shape_2d();
        $shape = [$shape, $shape];
        $check_sparse_nd_elemwise_binary->($shape, ['default', 'default'], $op, $g);
        $check_sparse_nd_elemwise_binary->($shape, ['row_sparse', 'row_sparse'], $op, $g);
    }
}

test_sparse_nd_elemwise_add();

sub test_sparse_nd_copy
{
    my $check_sparse_nd_copy = sub { my ($from_stype, $to_stype, $shape) = @_;
        my $from_nd = rand_ndarray($shape, $from_stype);
        # copy to ctx
        my $to_ctx = $from_nd->copyto(AI::MXNet::Context->current_ctx);
        # copy to stype
        my $to_nd = rand_ndarray($shape, $to_stype);
        $from_nd->copyto($to_nd);
        ok(($from_nd->aspdl != $to_ctx->aspdl)->abs->sum == 0);
        ok(($from_nd->aspdl != $to_nd->aspdl)->abs->sum == 0);
    };
    my $shape = rand_shape_2d();
    my $shape_3d = rand_shape_3d();
    my @stypes = ('row_sparse', 'csr');
    for my $stype (@stypes)
    {
        $check_sparse_nd_copy->($stype, 'default', $shape);
        $check_sparse_nd_copy->('default', $stype, $shape);
    }
    $check_sparse_nd_copy->('row_sparse', 'row_sparse', $shape_3d);
    $check_sparse_nd_copy->('row_sparse', 'default', $shape_3d);
    $check_sparse_nd_copy->('default', 'row_sparse', $shape_3d);
}

test_sparse_nd_copy();

sub test_sparse_nd_basic
{
    my $check_sparse_nd_basic_rsp = sub {
        my $storage_type = 'row_sparse';
        my $shape = rand_shape_2d();
        my ($nd) = rand_sparse_ndarray($shape, $storage_type);
        ok($nd->_num_aux == 1);
        ok($nd->indices->dtype eq 'int64');
        ok($nd->stype eq 'row_sparse');
    };
    $check_sparse_nd_basic_rsp->();
}

test_sparse_nd_basic();

sub test_sparse_nd_setitem
{
    my $check_sparse_nd_setitem = sub { my ($stype, $shape, $dst) = @_;
        my $x = mx->nd->zeros($shape, stype=>$stype);
        $x .= $dst;
        my $dst_nd = (blessed $dst and $dst->isa('PDL')) ? mx->nd->array($dst) : $dst;
        ok(($x->aspdl == (ref $dst_nd ? $dst_nd->aspdl : $dst_nd))->all);
    };

    my $shape = rand_shape_2d();
    for my $stype ('row_sparse', 'csr')
    {
        # ndarray assignment
        $check_sparse_nd_setitem->($stype, $shape, rand_ndarray($shape, 'default'));
        $check_sparse_nd_setitem->($stype, $shape, rand_ndarray($shape, $stype));
        # numpy assignment
        $check_sparse_nd_setitem->($stype, $shape, pones(reverse @{ $shape }));
    }
    # scalar assigned to row_sparse NDArray
    $check_sparse_nd_setitem->('row_sparse', $shape, 2);
}

test_sparse_nd_setitem();

sub test_sparse_nd_slice
{
    my $shape = [randint(2, 10), randint(2, 10)];
    my $stype = 'csr';
    my ($A) = rand_sparse_ndarray($shape, $stype);
    my $A2 = $A->aspdl;
    my $start = randint(0, $shape->[0] - 1);
    my $end = randint($start + 1, $shape->[0]);
    ok(same($A->slice([$start, $end])->aspdl, $A2->slice('X', [$start, $end])));
    ok(same($A->slice([$start - $shape->[0], $end])->aspdl, $A2->slice('X', [$start, $end])));
    ok(same($A->slice([$start, $shape->[0] - 1])->aspdl, $A2->slice('X', [$start, $shape->[0]-1])));
    ok(same($A->slice([0, $end])->aspdl, $A2->slice('X', [0, $end])));

    my $start_col = randint(0, $shape->[1] - 1);
    my $end_col = randint($start_col + 1, $shape->[1]);
    my $result = $A->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    my $result_dense = mx->nd->array($A2)->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    ok(same($result_dense->aspdl, $result->aspdl));

    $A = mx->nd->sparse->zeros('csr', $shape);
    $A2 = $A->aspdl;
    ok(same($A->slice([$start, $end])->aspdl, $A2->slice('X', [$start, $end])));
    $result = $A->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    $result_dense = mx->nd->array($A2)->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    ok(same($result_dense->aspdl, $result->aspdl));

    my $check_slice_nd_csr_fallback = sub { my ($shape) = @_;
        my $stype = 'csr';
        my ($A) = rand_sparse_ndarray($shape, $stype);
        my $A2 = $A->aspdl;
        my $start = randint(0, $shape->[0] - 1);
        my $end = randint($start + 1, $shape->[0]);

        # non-trivial step should fallback to dense slice op
        my $result = $A->slice(begin=>[$start], end=>[$end+1], step=>[2]);
        my $result_dense = mx->nd->array($A2)->slice(begin=>[$start], end=>[$end + 1], step=>[2]);
        ok(same($result_dense->aspdl, $result->aspdl));
    };
    $shape = [randint(2, 10), randint(1, 10)];
    $check_slice_nd_csr_fallback->($shape);
}

test_sparse_nd_slice();

sub test_sparse_nd_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x == $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 == $x;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_equal();

sub test_sparse_nd_not_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x != $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 != $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_not_equal();

sub test_sparse_nd_greater
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x > $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y > 0;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 > $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_greater();

sub test_sparse_nd_greater_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x >= $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y >= 0;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 >= $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y >= 1;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_greater_equal();

sub test_sparse_nd_lesser
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $y < $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 < $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = $y < 0;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_lesser();

sub test_sparse_nd_lesser_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $y <= $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 <= $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = $y <= 0;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 1 <= $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_lesser_equal();

sub test_sparse_nd_binary
{
    my $N = 2;
    my $check_binary = sub { my ($fn, $stype) = @_;
        for (0 .. 2)
        {
            my $ndim = 2;
            my $oshape = [map { randint(1, 6) } 1..$ndim];
            my $bdim = 2;
            my @lshape = @$oshape;
            # one for broadcast op, another for elemwise op
            my @rshape = @lshape[($ndim-$bdim)..@lshape-1];
            for my $i (0..$bdim-1)
            {
                my $sep = mx->nd->random->uniform(0, 1)->asscalar;
                if($sep < 0.33)
                {
                    $lshape[$ndim-$i-1] = 1;
                }
                elsif($sep < 0.66)
                {
                    $rshape[$bdim-$i-1] = 1;
                }
            }
            my $lhs = mx->nd->random->uniform(0, 1, shape=>\@lshape)->aspdl;
            my $rhs = mx->nd->random->uniform(0, 1, shape=>\@rshape)->aspdl;
            my $lhs_nd = mx->nd->array($lhs)->tostype($stype);
            my $rhs_nd = mx->nd->array($rhs)->tostype($stype);
            ok(allclose($fn->($lhs, $rhs), $fn->($lhs_nd, $rhs_nd)->aspdl, 1e-4));
        }
    };
    for my $stype ('row_sparse', 'csr')
    {
        $check_binary->(sub { $_[0] +  $_[1] }, $stype);
        $check_binary->(sub { $_[0] -  $_[1] }, $stype);
        $check_binary->(sub { $_[0] *  $_[1] }, $stype);
        $check_binary->(sub { $_[0] /  $_[1] }, $stype);
        $check_binary->(sub { $_[0] ** $_[1] }, $stype);
        $check_binary->(sub { $_[0] >  $_[1] }, $stype);
        $check_binary->(sub { $_[0] <  $_[1] }, $stype);
        $check_binary->(sub { $_[0] >= $_[1] }, $stype);
        $check_binary->(sub { $_[0] <= $_[1] }, $stype);
        $check_binary->(sub { $_[0] == $_[1] }, $stype);
    }
}

test_sparse_nd_binary();

sub test_sparse_nd_binary_scalar_op
{
    my $N = 3;
    my $check = sub { my ($fn, $stype) = @_;
        for (1..$N)
        {
            my $ndim = 2;
            my $shape = [map { randint(1, 6) } 1..$ndim];
            my $npy = mx->nd->random->normal(0, 1, shape=>$shape)->aspdl;
            my $nd = mx->nd->array($npy)->tostype($stype);
            ok(allclose($fn->($npy), $fn->($nd)->aspdl, 1e-4));
        }
    };
    for my $stype ('row_sparse', 'csr')
    {
        $check->(sub { 1 +    $_[0] }, $stype);
        $check->(sub { 1 -    $_[0] }, $stype);
        $check->(sub { 1 *    $_[0] }, $stype);
        $check->(sub { 1 /    $_[0] }, $stype);
        $check->(sub { 2 **   $_[0] }, $stype);
        $check->(sub { 1 >    $_[0] }, $stype);
        $check->(sub { 0.5 >  $_[0] }, $stype);
        $check->(sub { 0.5 <  $_[0] }, $stype);
        $check->(sub { 0.5 >= $_[0] }, $stype);
        $check->(sub { 0.5 <= $_[0] }, $stype);
        $check->(sub { 0.5 == $_[0] }, $stype);
        $check->(sub { $_[0] / 2    }, $stype);
    }
}

test_sparse_nd_binary_scalar_op();

sub test_sparse_nd_binary_iop
{
    my $N = 3;
    my $check_binary = sub { my ($fn, $stype) = @_;
        for (1..$N)
        {
            my $ndim = 2;
            my $oshape = [map { randint(1, 6) } 1..$ndim];
            my $lhs = mx->nd->random->uniform(0, 1, shape => $oshape)->aspdl;
            my $rhs = mx->nd->random->uniform(0, 1, shape => $oshape)->aspdl;
            my $lhs_nd = mx->nd->array($lhs)->tostype($stype);
            my $rhs_nd = mx->nd->array($rhs)->tostype($stype);
            ok(
                allclose(
                    $fn->($lhs, $rhs),
                    $fn->($lhs_nd, $rhs_nd)->aspdl,
                    1e-4
                )
            );
        }
    };

    my $inplace_add = sub { my ($x, $y) = @_;
        $x += $y;
        return $x
    };
    my $inplace_mul = sub { my ($x, $y) = @_;
        $x *= $y;
        return $x
    };
    my @stypes = ('csr', 'row_sparse');
    my @fns = ($inplace_add, $inplace_mul);
    for my $stype (@stypes)
    {
        for my $fn (@fns)
        {
            $check_binary->($fn, $stype);
        }
    }
}

test_sparse_nd_binary_iop();

sub test_sparse_nd_negate
{
    my $check_sparse_nd_negate = sub { my ($shape, $stype) = @_;
        my $npy = mx->nd->random->uniform(-10, 10, shape => rand_shape_2d())->aspdl;
        my $arr = mx->nd->array($npy)->tostype($stype);
        ok(almost_equal($npy, $arr->aspdl));
        ok(almost_equal(-$npy, (-$arr)->aspdl));

        # a final check to make sure the negation (-) is not implemented
        # as inplace operation, so the contents of arr does not change after
        # we compute (-arr)
        ok(almost_equal($npy, $arr->aspdl));
    };
    my $shape = rand_shape_2d();
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        $check_sparse_nd_negate->($shape, $stype);
    }
}

test_sparse_nd_negate();

sub test_sparse_nd_broadcast
{
    my $sample_num = 10; # TODO 1000
    my $test_broadcast_to = sub { my ($stype) = @_;
        for (1..$sample_num)
        {
            my $ndim = 2;
            my $target_shape = [map { randint(1, 11) } 1..$ndim];
            my $shape = \@{ $target_shape };
            my $axis_flags = [map { randint(0, 2) } 1..$ndim];
            my $axes = [];
            enumerate(sub {
                my ($axis, $flag) = @_;
                if($flag)
                {
                    $shape->[$axis] = 1;
                }
            }, $axis_flags);
            my $dat = mx->nd->random->uniform(0, 1, shape => $shape)->aspdl - 0.5;
            my $pdl_ret = $dat;
            my $ndarray = mx->nd->array($dat)->tostype($stype);
            my $ndarray_ret = $ndarray->broadcast_to($target_shape);
            ok((pdl($ndarray_ret->shape) == pdl($target_shape))->all);
            my $err = (($ndarray_ret->aspdl - $pdl_ret)**2)->avg;
            ok($err < 1E-8);
        }
    };
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        $test_broadcast_to->($stype);
    }
}

test_sparse_nd_broadcast();

sub test_sparse_nd_transpose
{
    my $npy = mx->nd->random->uniform(-10, 10, shape => rand_shape_2d())->aspdl;
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        my $nd = mx->nd->array($npy)->tostype($stype);
        ok(almost_equal($npy->transpose, ($nd->T)->aspdl));
    }
}

test_sparse_nd_transpose();

sub test_sparse_nd_storage_fallback
{
    my $check_output_fallback = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->zeros($shape, stype=>'csr');
        mx->nd->broadcast_add($ones, $ones * 2, out=>$out);
        ok(($out->aspdl - 3)->sum == 0);
    };

    my $check_input_fallback = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->broadcast_add($ones->tostype('csr'), $ones->tostype('row_sparse'));
        ok(($out->aspdl - 2)->sum == 0);
    };

    my $check_fallback_with_temp_resource = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->sum($ones);
        ok($out->asscalar == product(@{ $shape }));
    };

    my $shape = rand_shape_2d();
    $check_output_fallback->($shape);
    $check_input_fallback->($shape);
    $check_fallback_with_temp_resource->($shape);
}

test_sparse_nd_storage_fallback();

sub test_sparse_nd_astype
{
    my @stypes = ('row_sparse', 'csr');
    for my $stype (@stypes)
    {
        my $x = mx->nd->zeros(rand_shape_2d(), stype => $stype, dtype => 'float32');
        my $y = $x->astype('int32');
        ok($y->dtype eq 'int32');
    }
}

test_sparse_nd_astype();

sub test_sparse_nd_storable
{
    my $repeat = 1;
    my $dim0 = 40;
    my $dim1 = 40;
    my @stypes = ('row_sparse', 'csr');
    my @densities = (0, 0.5);
    my %stype = (row_sparse => 'AI::MXNet::NDArray::RowSparse', csr => 'AI::MXNet::NDArray::CSR');
    for (1..$repeat)
    {
        my $shape = rand_shape_2d($dim0, $dim1);
        for my $stype (@stypes)
        {
            for my $density (@densities)
            {
                my ($a) = rand_sparse_ndarray($shape, $stype, density => $density);
                ok($a->isa($stype{$stype}));
                my $data = Storable::freeze($a);
                my $b = Storable::thaw($data);
                ok($b->isa($stype{$stype}));
                ok(same($a->aspdl, $b->aspdl));
            }
        }
    }
}

test_sparse_nd_storable();

sub test_sparse_nd_save_load
{
    my $repeat = 1;
    my @stypes = ('default', 'row_sparse', 'csr');
    my %stype = (default => 'AI::MXNet::NDArray', row_sparse => 'AI::MXNet::NDArray::RowSparse', csr => 'AI::MXNet::NDArray::CSR');
    my $num_data = 20;
    my @densities = (0, 0.5);
    my $fname = 'tmp_list.bin';
    for (1..$repeat)
    {
        my @data_list1;
        for (1..$num_data)
        {
            my $stype = $stypes[randint(0, scalar(@stypes))];
            my $shape = rand_shape_2d(40, 40);
            my $density = $densities[randint(0, scalar(@densities))];
            push @data_list1, rand_ndarray($shape, $stype, $density);
            ok($data_list1[-1]->isa($stype{$stype}));
        }
        mx->nd->save($fname, \@data_list1);

        my @data_list2 = @{ mx->nd->load($fname) };
        ok(@data_list1 == @data_list2);
        zip(sub {
            my ($x, $y) = @_;
            ok(same($x->aspdl, $y->aspdl));
        }, \@data_list1, \@data_list2);

        my %data_map1;
        enumerate(sub {
            my ($i, $x) = @_;
            $data_map1{"ndarray xx $i"} = $x;
        }, \@data_list1);
        mx->nd->save($fname, \%data_map1);
        my %data_map2 = %{ mx->nd->load($fname) };
        ok(keys(%data_map1) == keys(%data_map2));
        while(my ($k, $x) = each %data_map1)
        {
            my $y = $data_map2{$k};
            ok(same($x->aspdl, $y->aspdl));
        }
    }
    unlink $fname;
}

test_sparse_nd_save_load();

sub test_create_csr
{
    my $check_create_csr_from_nd = sub { my ($shape, $density, $dtype) = @_;
        my $matrix = rand_ndarray($shape, 'csr', $density);
        # create data array with provided dtype and ctx
        my $data = mx->nd->array($matrix->data->aspdl, dtype=>$dtype);
        my $indptr = $matrix->indptr;
        my $indices = $matrix->indices;
        my $csr_created = mx->nd->sparse->csr_matrix([$data, $indices, $indptr], shape=>$shape);
        ok($csr_created->stype eq 'csr');
        ok(same($csr_created->data->aspdl, $data->aspdl));
        ok(same($csr_created->indptr->aspdl, $indptr->aspdl));
        ok(same($csr_created->indices->aspdl, $indices->aspdl));
        # verify csr matrix dtype and ctx is consistent from the ones provided
        ok($csr_created->dtype eq $dtype);
        ok($csr_created->data->dtype eq $dtype);
        ok($csr_created->context eq AI::MXNet::Context->current_ctx);
        my $csr_copy = mx->nd->array($csr_created);
        ok(same($csr_copy->aspdl, $csr_created->aspdl));
    };

    my $check_create_csr_from_coo = sub { my ($shape, $density, $dtype) = @_;
        my $matrix = rand_ndarray($shape, 'csr', $density);
        my $sp_csr = $matrix->aspdlccs;
        my $sp_coo = $sp_csr->tocoo();
        my $csr_created = mx->nd->sparse->csr_matrix([$sp_coo->data, [$sp_coo->row, $sp_coo->col]], shape=>$shape, dtype=>$dtype);
        ok($csr_created->stype eq 'csr');
        ok(same($csr_created->data->aspdl, $sp_csr->data));
        ok(same($csr_created->indptr->aspdl, $sp_csr->indptr));
        ok(same($csr_created->indices->aspdl, $sp_csr->indices));
        my $csr_copy = mx->nd->array($csr_created);
        ok(same($csr_copy->aspdl, $csr_created->aspdl));
        # verify csr matrix dtype and ctx is consistent
        ok($csr_created->dtype eq $dtype);
        ok($csr_created->data->dtype eq $dtype);
        ok($csr_created->context eq AI::MXNet::Context->current_ctx);
    };

    my $check_create_csr_from_pdlccs = sub { my ($shape, $density, $f) = @_;
        my $assert_csr_almost_equal = sub { my ($nd, $sp) = @_;
            ok(almost_equal($nd->data->aspdl, $sp->data));
            ok(almost_equal($nd->indptr->aspdl, $sp->indptr));
            ok(almost_equal($nd->indices->aspdl, $sp->indices));
            my $sp_csr = $nd->aspdlccs;
            ok(almost_equal($sp_csr->data, $sp->data));
            ok(almost_equal($sp_csr->indptr, $sp->indptr));
            ok(almost_equal($sp_csr->indices, $sp->indices));
            ok($sp->dtype eq $sp_csr->dtype);
        };

            my $csr_sp = rand_sparse($shape->[0], $shape->[1], $density);
            my $csr_nd = $f->($csr_sp);
            ok(almost_equal($csr_nd->aspdl, $csr_sp->todense));
            # non-canonical csr which contains duplicates and unsorted indices
            my $indptr = pdl([0, 2, 3, 7]);
            my $indices = pdl([0, 2, 2, 0, 1, 2, 1]);
            my $data = pdl([1, 2, 3, 4, 5, 6, 1]);
            my $non_canonical_csr = mx->nd->sparse->csr_matrix([$data, $indices, $indptr], shape=>[3, 3], dtype=>$csr_nd->dtype);
            my $canonical_csr_nd = $f->($non_canonical_csr, dtype=>$csr_nd->dtype);
            my $canonical_csr_sp = $non_canonical_csr->copy();
            ok(almost_equal($canonical_csr_nd->aspdl, $canonical_csr_sp->aspdl));
    };

    my $dim0 = 20;
    my $dim1 = 20;
    my @densities = (0.5);
    my $dtype = 'float64';
    for my $density (@densities)
    {
        my $shape = [$dim0, $dim1];
        $check_create_csr_from_nd->($shape, $density, $dtype);
        $check_create_csr_from_coo->($shape, $density, $dtype);
        $check_create_csr_from_pdlccs->($shape, $density, sub { mx->nd->sparse->array(@_) });
        $check_create_csr_from_pdlccs->($shape, $density, sub { mx->nd->array(@_) });
    }
}

test_create_csr();

sub test_create_row_sparse
{
    my $dim0 = 50;
    my $dim1 = 50;
    my @densities = (0, 0.5, 1);
    for my $density (@densities)
    {
        my $shape = rand_shape_2d($dim0, $dim1);
        my $matrix = rand_ndarray($shape, 'row_sparse', $density);
        my $data = $matrix->data;
        my $indices = $matrix->indices;
        my $rsp_created = mx->nd->sparse->row_sparse_array([$data, $indices], shape=>$shape);
        ok($rsp_created->stype eq 'row_sparse');
        ok(same($rsp_created->data->aspdl, $data->aspdl));
        ok(same($rsp_created->indices->aspdl, $indices->aspdl));
        my $rsp_copy = mx->nd->array($rsp_created);
        ok(same($rsp_copy->aspdl, $rsp_created->aspdl));
    }
}

test_create_row_sparse();

sub test_create_sparse_nd_infer_shape
{
    my $check_create_csr_infer_shape = sub { my ($shape, $density, $dtype) = @_;
        eval {
            my $matrix = rand_ndarray($shape, 'csr', $density);
            my $data = $matrix->data;
            my $indptr = $matrix->indptr;
            my $indices = $matrix->indices;
            my $nd = mx->nd->sparse->csr_matrix([$data, $indices, $indptr], dtype=>$dtype);
            my ($num_rows, $num_cols) = @{ $nd->shape };
            ok($num_rows == @{ $indptr } - 1);
            ok($indices->shape->[0] > 0);
            ok(($num_cols <= $indices)->aspdl->sum == 0);
            ok($nd->dtype eq $dtype);
        };
    };
    my $check_create_rsp_infer_shape = sub { my ($shape, $density, $dtype) = @_;
        eval {
            my $array = rand_ndarray($shape, 'row_sparse', $density);
            my $data = $array->data;
            my $indices = $array->indices;
            my $nd = mx->nd->sparse->row_sparse_array([$data, $indices], dtype=>$dtype);
            my $inferred_shape = $nd->shape;
            is_deeply([@{ $inferred_shape }[1..@{ $inferred_shape }-1]], [@{ $data->shape }[1..@{ $data->shape }-1]]);
            ok($indices->ndim > 0);
            ok($nd->dtype eq $dtype);
            if($indices->shape->[0] > 0)
            {
                ok(($inferred_shape->[0] <= $indices)->aspdl->sum == 0);
            }
        };
    };

    my $dtype = 'int32';
    my $shape = rand_shape_2d();
    my $shape_3d = rand_shape_3d();
    my @densities = (0, 0.5, 1);
    for my $density (@densities)
    {
        $check_create_csr_infer_shape->($shape, $density, $dtype);
        $check_create_rsp_infer_shape->($shape, $density, $dtype);
        $check_create_rsp_infer_shape->($shape_3d, $density, $dtype);
    }
}

test_create_sparse_nd_infer_shape();

sub test_create_sparse_nd_from_dense
{
    my $check_create_from_dns = sub { my ($shape, $f, $dense_arr, $dtype, $default_dtype, $ctx) = @_;
        my $arr = $f->($dense_arr, shape => $shape, dtype => $dtype, ctx => $ctx);
        ok(same($arr->aspdl, pones(reverse @{ $shape })));
        ok($arr->dtype eq $dtype);
        ok($arr->context eq $ctx);
        # verify the default dtype inferred from dense arr
        my $arr2 = $f->($dense_arr);
        ok($arr2->dtype eq $default_dtype);
        ok($arr2->context eq AI::MXNet::Context->current_ctx);
    };
    my $shape = rand_shape_2d();
    my $dtype = 'int32';
    my $src_dtype = 'float64';
    my $ctx = mx->cpu(1);
    my @dense_arrs = (
        mx->nd->ones($shape, dtype=>$src_dtype),
        mx->nd->ones($shape, dtype=>$src_dtype)->aspdl
    );
    for my $f (sub { mx->nd->sparse->csr_matrix(@_) }, sub { mx->nd->sparse->row_sparse_array(@_) })
    {
        for my $dense_arr (@dense_arrs)
        {
            my $default_dtype = blessed($dense_arr) ? $dense_arr->dtype : 'float32';
            $check_create_from_dns->($shape, $f, $dense_arr, $dtype, $default_dtype, $ctx);
        }
    }
}

test_create_sparse_nd_from_dense();

sub test_create_sparse_nd_from_sparse
{
    my $check_create_from_sp = sub { my ($shape, $f, $sp_arr, $dtype, $src_dtype, $ctx) = @_;
        my $arr = $f->($sp_arr, shape => $shape, dtype=>$dtype, ctx=>$ctx);
        ok(same($arr->aspdl, pones(reverse @{ $shape })));
        ok($arr->dtype eq $dtype);
        ok($arr->context eq $ctx);
        # verify the default dtype inferred from sparse arr
        my $arr2 = $f->($sp_arr);
        ok($arr2->dtype eq $src_dtype);
        ok($arr2->context eq AI::MXNet::Context->current_ctx);
    };

    my $shape = rand_shape_2d();
    my $src_dtype = 'float64';
    my $dtype = 'int32';
    my $ctx = mx->cpu(1);
    my $ones = mx->nd->ones($shape, dtype=>$src_dtype);
    my @csr_arrs = ($ones->tostype('csr'));
    my @rsp_arrs = ($ones->tostype('row_sparse'));
    push @csr_arrs, mx->nd->ones($shape, dtype=>$src_dtype)->aspdl->tocsr;
    my $f_csr = sub { mx->nd->sparse->csr_matrix(@_) };
    my $f_rsp = sub { mx->nd->sparse->row_sparse_array(@_) };
    for my $sp_arr (@csr_arrs)
    {
        $check_create_from_sp->($shape, $f_csr, $sp_arr, $dtype, $src_dtype, $ctx);
    }
    for my $sp_arr (@rsp_arrs)
    {
        $check_create_from_sp->($shape, $f_rsp, $sp_arr, $dtype, $src_dtype, $ctx);
    }
}

test_create_sparse_nd_from_sparse();

sub test_create_sparse_nd_empty
{
    my $check_empty = sub { my ($shape, $stype) = @_;
        my $arr = mx->nd->sparse->empty($stype, $shape);
        ok($arr->stype eq $stype);
        ok(same($arr->aspdl, pzeros(reverse(@{ $shape }))));
    };

    my $check_csr_empty = sub { my ($shape, $dtype, $ctx) = @_;
        my $arr = mx->nd->sparse->csr_matrix(undef, shape => $shape, dtype => $dtype, ctx => $ctx);
        ok($arr->stype eq 'csr');
        ok($arr->dtype eq $dtype);
        ok($arr->context eq $ctx);
        ok(same($arr->aspdl, pzeros(reverse(@{ $shape }))));
        # check the default value for dtype and ctx
        $arr = mx->nd->sparse->csr_matrix(undef, shape => $shape);
        ok($arr->dtype eq 'float32');
        ok($arr->context eq AI::MXNet::Context->current_ctx);
    };

    my $check_rsp_empty = sub { my ($shape, $dtype, $ctx) = @_;
        my $arr = mx->nd->sparse->row_sparse_array(undef, shape => $shape, dtype=>$dtype, ctx=>$ctx);
        ok($arr->stype eq 'row_sparse');
        ok($arr->dtype eq $dtype);
        ok($arr->context eq $ctx);
        ok(same($arr->aspdl, pzeros(reverse(@{ $shape }))));
        # check the default value for dtype and ctx
        $arr = mx->nd->sparse->row_sparse_array(undef, shape => $shape);
        ok($arr->dtype eq 'float32');
        ok($arr->context eq AI::MXNet::Context->current_ctx);
    };

    my @stypes = ('csr', 'row_sparse');
    my $shape = rand_shape_2d();
    my $shape_3d = rand_shape_3d();
    my $dtype = 'int32';
    my $ctx = mx->cpu(1);
    for my $stype (@stypes)
    {
        $check_empty->($shape, $stype);
    }
    $check_csr_empty->($shape, $dtype, $ctx);
    $check_rsp_empty->($shape, $dtype, $ctx);
    $check_rsp_empty->($shape_3d, $dtype, $ctx);
}

test_create_sparse_nd_empty();

sub test_synthetic_dataset_generator
{
    my $test_powerlaw_generator = sub { my ($csr_arr, $final_row) = @_;
        my $indices = $csr_arr->indices->aspdl;
        my $indptr = $csr_arr->indptr->aspdl;
        for my $row (1..$final_row)
        {
            my $nextrow = $row + 1;
            my $current_row_nnz = $indices->at($indptr->at($row) - 1) + 1;
            my $next_row_nnz = $indices->at($indptr->at($nextrow) - 1) + 1;
            ok($next_row_nnz == 2 * $current_row_nnz);
        }
    };

    # Test if density is preserved
    my ($csr_arr_cols) = rand_sparse_ndarray([32, 10000], "csr",
                                          density=>0.01, distribution=>"powerlaw");

    my ($csr_arr_small) = rand_sparse_ndarray([5, 5], "csr",
                                           density=>0.5, distribution=>"powerlaw");

    my ($csr_arr_big) = rand_sparse_ndarray([32, 1000000], "csr",
                                         density=>0.4, distribution=>"powerlaw");

    my ($csr_arr_square) = rand_sparse_ndarray([1600, 1600], "csr",
                                            density=>0.5, distribution=>"powerlaw");
    ok($csr_arr_cols->data->len == 3200);
    $test_powerlaw_generator->($csr_arr_cols, 9);
    $test_powerlaw_generator->($csr_arr_small, 1);
    $test_powerlaw_generator->($csr_arr_big, 4);
    $test_powerlaw_generator->($csr_arr_square, 6);
}

test_synthetic_dataset_generator();

sub test_sparse_nd_fluent
{
    my $check_fluent_regular = sub { my ($stype, $func, $kwargs, $shape, $equal_nan) = @_;
        $shape //= [5, 17];
        my $data = mx->nd->random->uniform(shape=>$shape)->tostype($stype);
        my $regular = AI::MXNet::NDArray::Base->$func($data, %$kwargs);
        my $fluent  = $data->$func(%$kwargs);
        ok(almost_equal($regular->aspdl, $fluent->aspdl));
    };

    my @common_func = ('zeros_like', 'square');
    my @rsp_func = ('round', 'rint', 'fix', 'floor', 'ceil', 'trunc',
                'abs', 'sign', 'sin', 'degrees', 'radians', 'expm1');
    for my $func (@common_func)
    {
        $check_fluent_regular->('csr', $func, {});
    }
    for my $func (@common_func, @rsp_func)
    {
        $check_fluent_regular->('row_sparse', $func, {});
    }

    @rsp_func = ('arcsin', 'arctan', 'tan', 'sinh', 'tanh',
                'arcsinh', 'arctanh', 'log1p', 'sqrt', 'relu');
    for my $func (@rsp_func)
    {
        $check_fluent_regular->('row_sparse', $func, {});
    }

    $check_fluent_regular->('csr', 'slice', {begin => [2, 5], end => [4, 7]});
    $check_fluent_regular->('row_sparse', 'clip', {a_min => -0.25, a_max => 0.75});

    for my $func ('sum', 'mean')
    {
        $check_fluent_regular->('csr', $func, {axis => 0});
    }
}

test_sparse_nd_fluent();

sub test_sparse_nd_exception
{
    my $a = mx->nd->ones([2,2]);
    dies_ok(sub { mx->nd->sparse->retain($a, invalid_arg=>"garbage_value") });
    dies_ok(sub { mx->nd->sparse->csr_matrix($a, shape=>[3,2]) });
    dies_ok(sub { mx->nd->sparse->csr_matrix(pdl([2,2]), shape=>[3,2]) });
    dies_ok(sub { mx->nd->sparse->row_sparse_array(pdl([2,2]), shape=>[3,2]) });
    dies_ok(sub { mx->nd->sparse->zeros("invalid_stype", [2,2]) });
}

test_sparse_nd_exception();

sub test_sparse_nd_check_format
{
    my $shape = rand_shape_2d();
    my @stypes = ("csr", "row_sparse");
    for my $stype (@stypes)
    {
        my ($arr) = rand_sparse_ndarray($shape, $stype);
        $arr->check_format();
        $arr = mx->nd->sparse->zeros($stype, $shape);
        $arr->check_format();
    }
    # CSR format index pointer array should be less than the number of rows
    $shape = [3, 4];
    my $data_list = [7, 8, 9];
    my $indices_list = [0, 2, 1];
    my $indptr_list = [0, 5, 2, 3];
    my $a = mx->nd->sparse->csr_matrix([$data_list, $indices_list, $indptr_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # CSR format indices should be in ascending order per row
    $indices_list = [2, 1, 1];
    $indptr_list = [0, 2, 2, 3];
    $a = mx->nd->sparse->csr_matrix([$data_list, $indices_list, $indptr_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # CSR format indptr should end with value equal with size of indices
    $indices_list = [1, 2, 1];
    $indptr_list = [0, 2, 2, 4];
    $a = mx->nd->sparse->csr_matrix([$data_list, $indices_list, $indptr_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # CSR format indices should not be negative
    $indices_list = [0, 2, 1];
    $indptr_list = [0, -2, 2, 3];
    $a = mx->nd->sparse->csr_matrix([$data_list, $indices_list, $indptr_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # Row Sparse format indices should be less than the number of rows
    $shape = [3, 2];
    $data_list = [[1, 2], [3, 4]];
    $indices_list = [1, 4];
    $a = mx->nd->sparse->row_sparse_array([$data_list, $indices_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # Row Sparse format indices should be in ascending order
    $indices_list = [1, 0];
    $a = mx->nd->sparse->row_sparse_array([$data_list, $indices_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
    # Row Sparse format indices should not be negative
    $indices_list = [1, -2];
    $a = mx->nd->sparse->row_sparse_array([$data_list, $indices_list], shape=>$shape);
    dies_ok(sub { $a->check_format });
}

test_sparse_nd_check_format();
