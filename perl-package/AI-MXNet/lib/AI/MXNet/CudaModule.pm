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

package AI::MXNet::CudaModule;
use strict;
use warnings;
use AI::MXNet::Base;
use Mouse;
use AI::MXNet::Function::Parameters;

our %DTYPE_CPP_TO_STR = qw(
    float    float32
    double   float64
    __half   float16
    uint8_t  uint8
    int      int32
    int32_t  int32
    int8_t   int8
    char     int8
    int64_t  int64
);

=head1 DESCRIPTION

    Interface to runtime cuda kernel compile module.
    Compile and run CUDA code from Perl.

    In CUDA 7.5, you need to prepend your kernel definitions
    with 'extern "C"' to avoid name mangling::

        $source = '
        extern "C" __global__ void axpy(const float *x, float *y, float alpha) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            y[i] += alpha * x[i];
        }
        ';
        $module = mx->rtc->CudaModule(source);
        $func = $module->get_kernel("axpy", "const float *x, float *y, float alpha");
        $x = mx->nd->ones([10]), ctx=>mx->gpu(0));
        $y = mx->nd->zeros([10]), ctx=>mx->gpu(0));
        $func->launch([$x, $y, 3.0], mx->gpu(0), [1, 1, 1], [10, 1, 1]);
        print $y->aspdl;

    Starting from CUDA 8.0, you can instead export functions by name.
    This also allows you to use templates::

        my $source = '
        template<typename DType>
        __global__ void axpy(const DType *x, DType *y, DType alpha) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            y[i] += alpha * x[i];
        }
        ';
        $module = mx->rtc->CudaModule($source, exports=>['axpy<float>', 'axpy<double>']);
        $func32 = $module->get_kernel("axpy<float>", "const float *x, float *y, float alpha");
        $x = mx->nd->ones([10], dtype=>'float32', ctx=>mx->gpu(0));
        $y = mx->nd->zeros([10], dtype=>'float32', ctx=>mx->gpu(0));
        $func32->launch([$x, $y, 3.0], mx->gpu(0), [1, 1, 1], [10, 1, 1]);
        print $y->aspdl;

        $func64 = $module->get_kernel("axpy<double>", "const double *x, double *y, double alpha");
        $x = mx->nd->ones([10], dtype=>'float64', ctx=>mx->gpu(0));
        $y = mx->nd->zeros([10], dtype=>'float64', ctx=>mx->gpu(0));
        $func32->launch([$x, $y, 3.0], mx->gpu(0), [1, 1, 1], [10, 1, 1]);
        print $y->aspdl;


    Parameters
    ----------
    source : str
        Complete source code.
    options : array ref of str
        Compiler flags. For example, use "-I/usr/local/cuda/include" to
        add cuda headers to include path.
    exports : array ref of str
        Export kernel names.
=cut

has 'source' => (is => 'rw', isa => 'Str', required => 1);
has [qw/options exports/] => (is => 'rw', isa => 'Str|ArrayRef[Str]', default => sub { [] });
has 'handle' => (is => 'rw', isa => 'CudaModuleHandle');
around BUILDARGS => \&AI::MXNet::Base::process_arguments;
method python_constructor_arguments() { ['source', 'options', 'exports'] }

sub BUILD
{
    my $self = shift;
    $self->options([$self->options]) unless ref $self->options;
    $self->options([$self->exports]) unless ref $self->exports;
    my $handle = check_call(
                    AI::MXNetCAPI::RtcCudaModuleCreate(
                        $self->source,
                        scalar(@{ $self->options }),
                        $self->options,
                        scalar(@{ $self->exports }),
                        $self->exports
                    )
    );
    $self->handle($handle);
}

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::RtcCudaModuleFree(shift->handle));
}

=head2 get_kernel

        Get CUDA kernel from compiled module.

        Parameters
        ----------
        name : str
            String name of the kernel.
        signature : str
            Function signature for the kernel. For example, if a kernel is
            declared as::

                extern "C" __global__ void axpy(const float *x, double *y, int alpha)

            Then its signature should be::

                const float *x, double *y, int alpha

            or::

                const float *, double *, int

            Note that `*` in signature marks an argument as array and
            `const` marks an argument as constant (input) array.

        Returns
        -------
        AI::MXNet::CudaKernel
            CUDA kernels that can be launched on GPUs.
=cut

method get_kernel(Str $name, Str $signature)
{
    my @is_ndarray;
    my @is_const;
    my @dtypes;
    my $pattern = qr/^\s*(const)?\s*([\w_]+)\s*(\*)?\s*([\w_]+)?\s*$/;
    $signature =~ s/\s+/ /g;
    my @args = split(/,/, $signature);
    for my $arg (@args)
    {
        if(not $arg =~ $pattern or $2 eq 'const')
        {
            confess(
                "Invalid function prototype \"$arg\". Must be in the ".
                'form of "(const) type (*) (name)'
            );
        }
        push @is_const, $1 ? 1 : 0;
        my $dtype = $2;
        push @is_ndarray, $3 ? 1 : 0;
        if(not exists $DTYPE_CPP_TO_STR{$dtype})
        {
            my $types = join(',', sort keys %DTYPE_CPP_TO_STR);
            confess("Unsupported kernel argument type $arg. Supported types are: $types.");
        }
        push @dtypes, DTYPE_STR_TO_MX->{$DTYPE_CPP_TO_STR{$dtype}};
    }

    my $handle = check_call(
        AI::MXNetCAPI::RtcCudaKernelCreate(
            $self->handle,
            $name,
            scalar(@dtypes),
            \@is_ndarray,
            \@is_const,
            \@dtypes
        )
    );
    return AI::MXNet::CudaKernel->new($handle, $name, \@is_ndarray, \@dtypes);
}

package AI::MXNet::CudaKernel;
use Mouse;
use AI::MXNet::Base;

=head1 NAME

    AI::MXNet::CudaKernel
=cut

=head1 DESCRIPTION

    Constructs CUDA kernel.
    Intended to be created by calling AI::MXNet::CudaModule->get_kernel only.
=cut

has [qw/handle name is_ndarray dtypes/] => (is => 'rw');
around BUILDARGS => sub {
    my ($orig, $class, $handle, $name, $is_ndarray, $dtypes) = @_;
    return $class->$orig(handle => $handle, name => $name, is_ndarray => $is_ndarray, dtypes => $dtypes);
};

sub BUILD
{
    my $self = shift;
    $self->dtypes([map { DTYPE_MX_TO_STR->{$_} } @{ $self->dtypes }]);
}

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::RtcCudaKernelFree(shift->handle));
}

=head2 launch

        Launch cuda kernel.

        Parameters
        ----------
        $args : array ref of NDArray or numbers
            List of arguments for kernel. NDArrays are expected for pointer
            types (e.g. `float*`, `double*`) while numbers are expected for
            non-pointer types (e.g. `int`, `float`).
        $ctx : AI::MXNet::Context
            The context to launch kernel on. Must be GPU context.
        $grid_dims : array ref of 3 integers
            Grid dimensions for CUDA kernel.
        $block_dims : array ref of 3 integers
            Block dimensions for CUDA kernel.
        $shared_mem=0 : integer, optional
            Size of dynamically allocated shared memory. Defaults to 0.
=cut

method launch(
    ArrayRef[AI::MXNet::NDArray|Num] $args,
    AI::MXNet::Context $ctx,
    CudaKernelShape $grid_dims,
    CudaKernelShape $block_dims,
    Int $shared_mem=0
)
{
    assert(($ctx->device_type eq 'gpu'), "Cuda kernel can only be launched on GPU");
    confess("CudaKernel(${\ $self->name }) expects ".scalar(@{$self->dtypes}). "arguments but got ".scalar(@$args).".")
        unless (@{ $args } == @{ $self->dtypes });
    my @void_args;
    enumerate(sub {
        my ($i, $arg, $is_nd, $dtype) = @_;
        if($is_nd)
        {
            confess("The $i-th argument is expected to be a NDArray but got [$arg]")
                unless blessed $arg;
            push @void_args, $arg->handle;
        }
        else
        {
            my $perl_pack_type = DTYPE_MX_TO_PERL->{$dtype};
            my $packed_arg;
            ## special handling for float16
            if($perl_pack_type eq 'S')
            {
                $packed_arg = pack("S", AI::MXNetCAPI::_float_to_half($arg));
            }
            else
            {
                $packed_arg = pack($perl_pack_type, $arg);

            }
            push @void_args, $packed_arg;
        }
    }, $args, $self->is_ndarray, $self->dtypes);
    check_call(
        AI::MXNetCAPI::RtcCudaKernelCall(
            $self->handle,
            $ctx->device_id,
            \@void_args,
            @{ $grid_dims },
            @{ $block_dims },
            $shared_mem
        )
    );
}

1;