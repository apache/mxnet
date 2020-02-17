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

use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(dies_ok);
use Test::More 'no_plan';
use Scalar::Util qw(refaddr);

sub test_features
{
    my $features = mx->runtime->Features();
    ok(exists $features->features->{CUDA});
    ok(keys %{ $features->features } >= 30);
}

sub test_is_singleton
{
    my $x = mx->runtime->Features();
    my $y = mx->runtime->Features();
    ok(refaddr($x) == refaddr($y));
}

sub test_is_enabled
{
    my $features = mx->runtime->Features();
    for my $f (keys %{ $features->features })
    {
        if($features->features->{$f})
        {
           ok($features->is_enabled($f));
        }
        else
        {
           ok(not $features->is_enabled($f));
        }
    }
}

sub test_is_enabled_not_existing
{
    my $features = mx->runtime->Features();
    dies_ok(sub { $features->is_enabled("hello world") });
}

test_features();
test_is_singleton();
test_is_enabled();
test_is_enabled_not_existing();
