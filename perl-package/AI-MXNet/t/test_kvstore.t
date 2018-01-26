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
use Test::More tests => 38;
use AI::MXNet qw(mx);

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
