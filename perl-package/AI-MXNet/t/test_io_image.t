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
use Test::More tests => 1;
use AI::MXNet qw(mx);
use Time::HiRes qw(time);

sub run_imageiter
{
    my ($path_rec, $n, $batch_size) = @_;
    $batch_size //= 32;
    my $data = mx->img->ImageIter(
        batch_size=>$batch_size,
        data_shape=>[3, 224, 224],
        path_imgrec=>$path_rec,
        kwargs => { rand_crop=>1,
        rand_resize=>1,
        rand_mirror=>1 }
    );
    $data->reset();
    my $tic = time;
    for my $i (1..$n)
    {
        $data->next;
        mx->nd->waitall;
        warn("average speed after iteration $i is " . $batch_size*$i/(time - $tic) . " samples/sec");
    }
}

run_imageiter('data/cifar/test.rec', 20);
ok(1);