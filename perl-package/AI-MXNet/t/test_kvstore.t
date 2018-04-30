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
use Test::More tests => 62;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(almost_equal randint zip rand_ndarray);
use AI::MXNet::Base qw(pzeros);

my $shape = [4, 4];
my $keys  = [5,7,9];

sub init_kv
{
    # init kv
    my $kv = mx->kv->create();
    # single
    $kv->init(3, mx->nd->zeros($shape));
    # list
    $kv->init($keys, [map { mx->nd->zeros($shape) } 0..@$keys-1]);
    return $kv;
}

sub check_diff_to_scalar
{
    # assert A == x
    my ($A, $x) = @_;
    ok(($A - $x)->aspdl->abs->sum == 0);
}

sub test_single_kv_pair
{
    # single key-value pair push & pull
    my $kv = init_kv();
    $kv->push(3, mx->nd->ones($shape));
    my $val = mx->nd->empty($shape);
    $kv->pull(3, out => $val);
    check_diff_to_scalar($val, 1);
}

sub test_init
{
    my $kv = mx->kv->create();
    $kv->init(3, mx->nd->ones($shape)*4);
    my $a = mx->nd->zeros($shape);
    $kv->pull(3, out=>$a);
    check_diff_to_scalar($a, 4);
}

sub test_list_kv_pair
{
    # list key-value pair push & pull
    my $kv = init_kv();
    $kv->push($keys, [map {mx->nd->ones($shape)*4} 0..@$keys-1]);
    my $val = [map { mx->nd->empty($shape) } 0..@$keys-1];
    $kv->pull($keys, out => $val);
    for my $v (@$val)
    {
        check_diff_to_scalar($v, 4);
    }
}

sub test_row_sparse_pull
{
    my $kv = mx->kv->create();
    $kv->init('e', mx->nd->ones($shape)->tostype('row_sparse'));

    my $check_row_sparse_pull = sub { my ($kv, $count) = @_;
        my $num_rows = $shape->[0];
        my $vals = [];
        my $row_ids = [];
        my $all_row_ids = mx->nd->array([0..$num_rows-1])->aspdl;
        for my $i (0..$count-1)
        {
            push @$vals, mx->nd->zeros($shape)->tostype('row_sparse');
            my $row_id = [map { randint(0, $num_rows) } 1..$num_rows];
            push @$row_ids, mx->nd->array($row_id)->reshape([2, int($num_rows/2)]);
        }
        my $row_ids_to_pull = @$row_ids == 1 ? $row_ids->[0] : $row_ids;
        my $vals_to_pull = @$vals == 1 ? $vals->[0] : $vals;

        $kv->row_sparse_pull('e', out=>$vals_to_pull, row_ids=>$row_ids_to_pull);
        zip(sub {
            my ($val, $row_id) = @_;
            my $retained_val = $val->aspdl;
            my %excluded_row_ids = map { $_ => 1 } @{ PDL::setops($all_row_ids, 'XOR', $row_id->aspdl)->unpdl };
            for my $row (0..$num_rows-1)
            {
                my $expected_val = pzeros(@{ $retained_val->at($row)->shape->unpdl });
                $expected_val += exists $excluded_row_ids{ $row } ? 0 : 1;
                ok(almost_equal($retained_val->at($row), $expected_val));
            }
        }, $vals, $row_ids);
    };
    $check_row_sparse_pull->($kv, 1);
    $check_row_sparse_pull->($kv, 4);
}

sub test_sparse_aggregator
{
    my $stype = 'row_sparse';
    my $kv = mx->kv->create($stype);
    $kv->init('a', mx->nd->zeros($shape, stype=>$stype));
    $kv->init($keys, [map { mx->nd->zeros($shape, stype=>$stype) } 0..@$keys-1]);

    # devices
    my $num_devs = 4;
    my $devs = [map { mx->cpu($_) } 0..$num_devs];

    # single
    my $vals = [map { rand_ndarray($shape, $stype)->copyto($devs->[$_]) } 0..$num_devs-1];
    my $expected_sum = mx->nd->zeros($shape)->aspdl;
    for my $v (@$vals)
    {
        $expected_sum += $v->aspdl;
    }

    # prepare row_ids
    my $all_rows = mx->nd->array([0..$shape->[0]-1]);
    $kv->push('a', $vals);
    $kv->row_sparse_pull('a', out=>$vals, row_ids=>[($all_rows)x@$vals]);
    my $result_sum = mx->nd->zeros($shape)->aspdl;
    for my $v (@$vals)
    {
        $result_sum += $v->aspdl;
    }
    ok(almost_equal($result_sum, $expected_sum * $num_devs));

    # list
    $vals = [([map { rand_ndarray($shape, $stype)->copyto($devs->[$_]) } 0..$num_devs-1])x@$keys];
    $expected_sum = mx->nd->zeros($shape)->aspdl;
    for my $v (@{ $vals->[0] })
    {
        $expected_sum += $v->aspdl;
    }

    $kv->push($keys, $vals);
    $kv->row_sparse_pull($keys, out=>$vals, row_ids=>[([($all_rows)x$num_devs])x@$vals]);
    for my $vv (@$vals)
    {
        $result_sum = mx->nd->zeros($shape)->aspdl;
        for my $v (@$vv)
        {
            $result_sum += $v->aspdl;
        }
        ok(almost_equal($result_sum, $expected_sum * $num_devs))
    }
}

sub test_aggregator
{
    # aggregate value on muliple devices

    my $kv = init_kv();

    # devices
    my $num_devs = 4;
    my $devs = [map { mx->cpu($_) } 0..$num_devs-1];

    # single
    my $vals = [map { mx->nd->ones($shape, ctx => $_) } @$devs];

    $kv->push(3, $vals);
    $kv->pull(3, out => $vals);

    for my $v (@$vals)
    {
        check_diff_to_scalar($v, $num_devs);
    }
    # list

    $vals = [map { [map { mx->nd->ones($shape, ctx => $_)*2 } @$devs] } 0..@$keys-1];
    $kv->push($keys, $vals);
    $kv->pull($keys, out => $vals);

    for my $vv (@{ $vals })
    {
        for my $v (@{ $vv })
        {
            check_diff_to_scalar($v, $num_devs * 2);
        }
    }
}

sub updater
{
    my ($key, $recv, $local) = @_;
    $local += $recv;
}

sub test_updater
{
    my ($dev) = @_;
    $dev //= 'cpu';
    my $kv = init_kv();
    $kv->_set_updater(\&updater);

    # devices
    my $num_devs = 4;
    my $devs = [map { mx->$dev($_) } 0..$num_devs-1];

    # single
    my $vals = [map { mx->nd->ones($shape, ctx => $_) } @$devs];

    $kv->push(3, $vals);
    $kv->pull(3, out => $vals);

    for my $v (@$vals)
    {
        check_diff_to_scalar($v, $num_devs);
    }

    # list
    $vals = [map { [map { mx->nd->ones($shape, ctx => $_) } @$devs] } 0..@$keys-1];

    my $num_push = 10;
    for my $i (0..$num_push-1)
    {
        $kv->push($keys, $vals);
    }

    $kv->pull($keys, out => $vals);

    for my $vv (@{ $vals })
    {
        for my $v (@{ $vv })
        {
            check_diff_to_scalar($v, $num_devs * $num_push);
        }
    }
}

sub test_get_type
{
    my $kvtype = 'local_allreduce_cpu';
    my $kv = mx->kv->create($kvtype);
    is($kv->type, $kvtype);
}

test_init();
test_get_type();
test_single_kv_pair();
test_list_kv_pair();
test_aggregator();
test_updater();
test_row_sparse_pull();
test_sparse_aggregator();
