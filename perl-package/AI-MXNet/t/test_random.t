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
use Test::More tests => 505;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(same enumerate);

sub check_with_device
{
    my ($device, $dtype) = @_;
    my $tol = 0.1;
    my @symbols = (
        {
            name   => 'normal',
            symbol => sub { mx->sym->random->normal(@_) },
            ndop   => sub { mx->nd->random->normal(@_)  },
            params => { loc => 10.0, scale => 0.5 },
            inputs => [ [loc => [ [ 0.0, 2.5 ], [ -9.75, -7.0 ] ]] , [scale => [ [ 1.0, 3.7 ], [ 4.2, 1.5 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{loc} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - $params->{scale} }, $tol]
            ]
        },
        {
            name   => 'uniform',
            symbol => sub { mx->sym->random->uniform(@_) },
            ndop   => sub { mx->nd->random->uniform(@_)  },
            params => { low => -1.5, high => 3 },
            inputs => [ [low => [ [ 0.0, 2.5 ], [ -9.75, -1.0 ] ]] , [high => [ [ 1.0, 3.7 ], [ 4.2, 10.5 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - ($params->{low} + $params->{high})/2 }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - sqrt(1/12) * ($params->{high} - $params->{low}) }, $tol]
            ]
        },
        {
            name   => 'gamma',
            symbol => sub { mx->sym->random->gamma(@_) },
            ndop   => sub { mx->nd->random->gamma(@_)  },
            params => { alpha => 9, beta => 0.5 },
            inputs => [ [alpha => [ [ 0.0, 2.5 ], [ 9.75, 11 ] ]] , [beta => [ [ 1, 0.7 ], [ 0.5, 0.3 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{alpha} * $params->{beta} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - sqrt($params->{alpha} * $params->{beta}**2) }, $tol]
            ]
        },
        {
            name   => 'exponential',
            symbol => sub { mx->sym->random->exponential(@_) },
            ndop   => sub { mx->nd->random->exponential(@_)  },
            params => { scale => 1/4 },
            inputs => [ [scale => [ [ 1/1, 1/8.5 ], [ 1/2.7, 1/0.5 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{scale} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - $params->{scale} }, $tol]
            ]
        },
        {
            name   => 'poisson',
            symbol => sub { mx->sym->random->poisson(@_) },
            ndop   => sub { mx->nd->random->poisson(@_)  },
            params => { lam => 4 },
            inputs => [ [lam => [ [ 1, 8.5 ], [ 2.7, 0.5 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{lam} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - sqrt($params->{lam}) }, $tol]
            ]
        },
        {
            name   => 'neg-binomial',
            symbol => sub { mx->sym->random->negative_binomial(@_) },
            ndop   => sub { mx->nd->random->negative_binomial(@_)  },
            params => { k => 3, p => 0.4 },
            inputs => [ [k => [ [ 3, 4 ], [ 5, 6 ] ]] , [p => [ [ 0.4, 0.77 ], [ 0.5, 0.84 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{k}*(1-$params->{p})/$params->{p} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - sqrt($params->{k}*(1-$params->{p}))/$params->{p} }, $tol]
            ]
        },
        {
            name   => 'gen-neg-binomial',
            symbol => sub { mx->sym->random->generalized_negative_binomial(@_) },
            ndop   => sub { mx->nd->random->generalized_negative_binomial(@_)  },
            params => { mu => 2, alpha => 0.3 },
            inputs => [ [mu => [ [ 2, 2.5 ], [ 1.3, 1.9 ] ]] , [alpha => [ [ 1.0, 0.1 ], [ 0.2, 0.5 ] ]] ],
            checks => [
                [mean => sub { my ($x, $params) = @_; $x->astype('float64')->aspdl->avg - $params->{mu} }, $tol],
                [std  => sub { my ($x, $params) = @_; ($x->astype('float64')->aspdl->stats)[6] - sqrt($params->{mu}+$params->{alpha}*$params->{mu}**2) }, $tol]
            ]
        },
    );
    my $shape = [1000, 1000];
    for my $symbdic (@symbols)
    {
        my $name = $symbdic->{name};
        my $ndop = $symbdic->{ndop};

        # check directly
        my %params = %{ $symbdic->{params} };
        %params = (%params, shape=>$shape, dtype=>$dtype, ctx=>$device);
        mx->random->seed(128);
        my $ret1 = $ndop->(%params);
        mx->random->seed(128);
        my $ret2 = $ndop->(%params);
        ok(same($ret1->aspdl, $ret2->aspdl), "simple $name");

        for my $d (@{ $symbdic->{checks} })
        {
            my ($check_name, $check_func, $tol) = @$d;
            ok((abs($check_func->($ret1, \%params)) < $tol), "simple $name, $check_name");
        }

        # check multi-distribution sampling, only supports cpu for now
        %params = (shape=>$shape, dtype=>$dtype, ctx=>$device);
        %params = (%params, map { $_->[0] => mx->nd->array($_->[1], ctx=>$device, dtype=>$dtype) } @{ $symbdic->{inputs} });
        mx->random->seed(128);
        $ret1 = $ndop->(%params);
        mx->random->seed(128);
        $ret2 = $ndop->(%params);
        ok(same($ret1->aspdl, $ret2->aspdl), "advanced $name");

        for my $i (0,1)
        {
            for my $j (0,1)
            {
                my %stats = map { $_->[0] => $_->[1][$i][$j] } @{ $symbdic->{inputs} };
                for my $d (@{ $symbdic->{checks} })
                {
                    my ($check_name, $check_func, $tol) = @$d;
                    ok((abs($check_func->($ret2->at($i)->at($j), \%stats)) < $tol), "advanced $name, $check_name");
                }
            }
        }

        # check symbolic
        my $symbol = $symbdic->{symbol};
        my $X = mx->sym->Variable("X");
        %params = %{ $symbdic->{params} };
        %params = (%params, shape=>$shape, dtype=>$dtype);
        my $Y = $symbol->(%params) + $X;
        my $x = mx->nd->zeros($shape, dtype=>$dtype, ctx=>$device);
        my $xgrad = mx->nd->zeros($shape, dtype=>$dtype, ctx=>$device);
        my $yexec = $Y->bind(ctx => $device, args => { X => $x }, args_grad => { X => $xgrad });
        mx->random->seed(128);
        $yexec->forward(1);
        $yexec->backward($yexec->outputs->[0]);
        my $un1 = ($yexec->outputs->[0] - $x)->copyto($device);
        ok(same($xgrad->aspdl, $un1->aspdl), "symbolic simple");
        mx->random->seed(128);
        $yexec->forward();
        my $un2 = ($yexec->outputs->[0] - $x)->copyto($device);
        ok(same($un1->aspdl, $un2->aspdl), "symbolic simple $name");

        for my $d (@{ $symbdic->{checks} })
        {
            my ($check_name, $check_func, $tol) = @$d;
            ok((abs($check_func->($un1, \%params)) < $tol), "symbolic $name, $check_name");
        }

        # check multi-distribution sampling, only supports cpu for now
        $symbol = $symbdic->{symbol};
        %params = (shape=>$shape, dtype=>$dtype);
        my $single_param = @{ $symbdic->{inputs} } == 1;
        my $v1 = mx->sym->Variable('v1');
        my $v2 = mx->sym->Variable('v2');
        $Y = $symbol->($single_param ? ($v1) : ($v1, $v2), %params);
        my $bindings = { v1 => mx->nd->array($symbdic->{inputs}[0][1]) };
        if(not $single_param)
        {
            $bindings->{v2} = mx->nd->array($symbdic->{inputs}[1][1]);
        }
        $yexec = $Y->bind(ctx=>$device, args=>$bindings);
        $yexec->forward();
        $un1 = $yexec->outputs->[0]->copyto($device);
        %params = ();
        enumerate(sub {
            my ($i, $r) = @_;
            enumerate(sub {
                my ($j, $p1) = @_;
                $params{ $symbdic->{inputs}[0][0] } = $p1;
                if(not $single_param)
                {
                    $params{ $symbdic->{inputs}[1][0] } = $symbdic->{inputs}[1][1][$i][$j];
                }
                my $samples = $un1->at($i)->at($j);
                for my $d (@{ $symbdic->{checks} })
                {
                    my ($check_name, $check_func, $tol) = @$d;
                    ok((abs($check_func->($samples, \%params)) < $tol), "symbolic advanced $name, $check_name");
                }
            }, $r);
        }, $symbdic->{inputs}[0][1]);
    }
}

sub test_random
{
    check_with_device(mx->context->current_context(), 'float16');
    check_with_device(mx->context->current_context(), 'float32');
    check_with_device(mx->context->current_context(), 'float64');
}

test_random();

sub test_sample_multinomial
{
    my $x = mx->nd->array([[0,1,2,3,4],[4,3,2,1,0]])/10.0;
    ok(@{ mx->nd->random->multinomial($x, shape=>1000, get_prob=>1) }, "multiminomial");
}

test_sample_multinomial();

