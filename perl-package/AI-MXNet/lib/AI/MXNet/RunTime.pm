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

package AI::MXNet::RunTime;
use strict;
use warnings;
use AI::MXNet::NS;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Mouse;

=encoding utf-8

=head1 NAME

    AI::MXNet::RunTime - Runtime querying of compile time features in the native library.
=cut

=head1 DESCRIPTION

    With this module you can check at runtime which libraries and features were compiled in the library.

    Example usage:

    use AI::MXNet qw(mx);
    my $features = mx->runtime->Features();

    print $features->is_enabled("CUDNN");
    0

    print $features->is_enabled("CPU_SSE");
    1

    print Dumper($features->features);
    $VAR1 = {
            'LAPACK' => 1,
            'F16C' => 1,
            'CPU_SSE2' => 1,
            'BLAS_MKL' => 0,
            'CXX14' => 0,
            'DIST_KVSTORE' => 0,
            'NCCL' => 0,
            'OPENMP' => 1,
            'CUDNN' => 0,
            'CPU_AVX' => 1,
            'CUDA_RTC' => 0,
            'BLAS_OPEN' => 1,
            'CPU_SSE4_2' => 1,
            'CPU_SSE4A' => 0,
            'TVM_OP' => 0,
            'MKLDNN' => 0,
            'TENSORRT' => 0,
            'JEMALLOC' => 1,
            'SSE' => 0,
            'PROFILER' => 0,
            'DEBUG' => 0,
            'BLAS_APPLE' => 0,
            'CPU_SSE3' => 1,
            'INT64_TENSOR_SIZE' => 0,
            'CPU_SSE4_1' => 1,
            'CUDA' => 0,
            'OPENCV' => 1,
            'CPU_SSE' => 1,
            'SIGNAL_HANDLER' => 0,
            'BLAS_ATLAS' => 0,
            'CAFFE' => 0,
            'CPU_AVX2' => 0
    };

    print $features;
    [✖ CUDA, ✖ CUDNN, ✖ NCCL, ✖ CUDA_RTC, ✖ TENSORRT, ✔ CPU_SSE, ✔ CPU_SSE2, ✔ CPU_SSE3,
    ✔ CPU_SSE4_1, ✔ CPU_SSE4_2, ✖ CPU_SSE4A, ✔ CPU_AVX, ✖ CPU_AVX2, ✔ OPENMP, ✖ SSE,
    ✔ F16C, ✔ JEMALLOC, ✔ BLAS_OPEN, ✖ BLAS_ATLAS, ✖ BLAS_MKL, ✖ BLAS_APPLE, ✔ LAPACK,
    ✖ MKLDNN, ✔ OPENCV, ✖ CAFFE, ✖ PROFILER, ✖ DIST_KVSTORE, ✖ CXX14, ✖ INT64_TENSOR_SIZE,
    ✔ SIGNAL_HANDLER, ✔ DEBUG, ✖ TVM_OP]

=cut
use overload '""' => sub {
    my $self = shift;
    my $s = join(', ', map {
        sprintf("%s %s", $self->features->{ $_ } ? '✔' : '✖', $_)
    } sort keys %{ $self->features });
    return "[$s]";
};

has 'features' => (is => 'rw', init_arg => undef, default => sub  {
    return scalar(check_call(AI::MXNetCAPI::LibInfoFeatures()));
});

method is_enabled(Str $feature)
{
    confess("Feature $feature does not exist")
        unless exists $self->features->{ $feature };
    return $self->features->{ $feature };
}

my $features;
method Features()
{
    $features //= __PACKAGE__->new;
    return $features;
}

1;
