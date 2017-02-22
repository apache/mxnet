package AI::MXNet::Rtc;
use strict;
use warnings;
use AI::MXNet::Base;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 DESCRIPTION

    Interface to runtime cuda kernel compile module.
=cut

=head2 Constructor

    MXRtc object in mxnet.
    This class allow you to write cuda kernel in perl
    and call them with NDArray.

    Parameters
    ----------
    name : str
        name of the kernel
    inputs : tuple of (str, mxnet.ndarray)
        list of input names and ndarray
    outputs : tuple of (str, mxnet.ndarray)
        list of output names and ndarray
    kernel : str
        the actual kernel code.
        Note that this is only the body of the kernel, i.e.
        after { and before }. Rtc will decorate the kernel.
        For example, if name = "mykernel" and
        inputs = [('x', mx.nd.zeros((10,)))]
        outputs = [('y', mx.nd.zeros((10,)))]
        kernel = "y[threadIdx.x] = x[threadIdx.x];",
        the kernel that is compile will be:
        extern "C" __global__ mykernel(float *x, float *y) {
            const int x_ndim = 1;
            const int x_dims = { 10 };
            const int y_ndim = 1;
            const int y_dims = { 10 };

            y[threadIdx.x] = x[threadIdx.x];
        }
=cut

has 'handle'              => (is => 'rw', isa => 'RtcHandle', init_arg => undef);
has [qw/name kernel/]     => (is => 'ro', isa => 'Str', required => 1);
has [qw/inputs outputs/]  => (is => 'ro', isa => 'HashRef[AI::MXNet::NDArray]', required => 1);

sub BUILD
{
    my $self = shift;
    my (@input_names, @output_names, @input_nds, @output_nds);
    while(my ($name, $arr) = each %{ $self->inputs })
    {
        push @input_names, $name;
        push @input_nds, $arr->handle;
    }
    while(my ($name, $arr) = each %{ $self->outputs })
    {
        push @output_names, $name;
        push @output_nds, $arr->handle;
    }
    my $handle = check_call(
        AI::MXNetCAPI::RtcCreate(
            $self->name,
            scalar(@input_names),
            scalar(@output_names),
            \@input_names,
            \@output_names,
            \@input_nds,
            \@output_nds,
            $self->kernel
        )
    );
    $self->handle($handle);
}

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::MXRtcFree(shift->handle));
}

=head2 push

        run the kernel.

        Parameters
        ----------
        inputs : list of ndarray
            list of input. Can be different ndarray then uses for constructor,
            but must have the same shape and in the same order.
        outputs : list of ndarray
            list of out. Can be different ndarray then uses for constructor,
            but must have the same shape and in the same order.
        grid_dims : tuple of 3 uint
            grid dimension for kernel launch
        block_dims : tuple of 3 uint
            block dimension for kernel launch
=cut


method push(
    ArrayRef[AI::MXNet::NDArray] $inputs,
    ArrayRef[AI::MXNet::NDArray] $outputs,
    ArrayRef[DimSize] $grid_dims,
    ArrayRef[DimSize] $block_dims
)
{
    confess("grid_dims must be size of 3")
        unless @{ $grid_dims } == 3;
    confess("block_dims must be size of 3")
        unless @{ $block_dims } == 3;
    check_call(
        AI::MXNetCAPI::RtcPush(
            $self->handle,
            scalar(@$inputs),
            scalar(@$outputs),
            [map { $_->handle } @$inputs],
            [map { $_->handle } @$outputs],
            @{ $grid_dims },
            @{ $block_dims }
        )
    );
}

1;