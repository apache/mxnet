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
use Test::More tests => 3;
my $gpu_present = (`perl -e 'use AI::MXNet qw(mx); print mx->nd->ones([1], ctx => mx->gpu(0))->asscalar' 2>/dev/null` eq '1');

sub test_cuda_rtc
{
    my $source = '
    extern "C" __global__ void axpy(const float *x, float *y, float alpha) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        y[i] += alpha * x[i];
    }

    extern "C" __global__ void saxpy(const float *x, float *y, float alpha) {
        extern __shared__ float smem[];
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        smem[threadIdx.x] = x[i];
        y[i] += alpha * smem[threadIdx.x];
    }
    ';
    my $module = mx->rtc->CudaModule($source);
    my $axpy = $module->get_kernel("axpy", "const float *x, float *y, float alpha");
    my $x = mx->nd->ones([10], ctx=>mx->gpu(0));
    my $y = mx->nd->zeros([10], ctx=>mx->gpu(0));
    $axpy->launch([$x, $y, 3], mx->gpu(0), [1, 1, 1], [10, 1, 1]);
    ok(($y->aspdl == 3)->all);

    my $saxpy = $module->get_kernel("saxpy", "const float *x, float *y, float alpha");
    $saxpy->launch([$x, $y, 4], mx->gpu(0), [1, 1, 1], [10, 1, 1], 10);
    ok(($y->aspdl == 7)->all);

    $saxpy->launch([$x, $y, 5], mx->gpu(0), [2, 1, 1], [5, 1, 1], 5);
    ok(($y->aspdl == 12)->all);
}

SKIP: {
    skip("GPU is not avalilable", 3) unless $gpu_present;
    test_cuda_rtc();
}