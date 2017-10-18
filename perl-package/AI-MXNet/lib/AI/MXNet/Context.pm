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

package AI::MXNet::Context;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Types;
use AI::MXNet::Function::Parameters;
use constant devtype2str => { 1 => 'cpu', 2 => 'gpu', 3 => 'cpu_pinned' };
use constant devstr2type => { cpu => 1, gpu => 2, cpu_pinned => 3 };
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(device_type => $_[0])
        if @_ == 1 and $_[0] =~ /^(?:cpu|gpu|cpu_pinned)$/;
    return $class->$orig(
        device_type => $_[0]->device_type,
        device_id   => $_[0]->device_id
    ) if @_ == 1 and blessed $_[0];
    return $class->$orig(device_type => $_[0], device_id => $_[0])
        if @_ == 2 and $_[0] =~ /^(?:cpu|gpu|cpu_pinned)$/;
    return $class->$orig(@_);
};

has 'device_type' => (
    is => 'rw',
    isa => enum([qw[cpu gpu cpu_pinned]]),
    default => 'cpu'
);

has 'device_type_id' => (
    is => 'rw',
    isa => enum([1, 2, 3]),
    default => sub { devstr2type->{ shift->device_type } },
    lazy => 1
);

has 'device_id' => (
    is => 'rw',
    isa => 'Int',
    default => 0
);

use overload
    '==' => sub {
        my ($self, $other) = @_;
        return 0 unless blessed($other) and $other->isa(__PACKAGE__);
        return "$self" eq "$other";
    },
    '""' => sub {
        my ($self) = @_;
        return sprintf("%s(%s)", $self->device_type, $self->device_id);
    },
    fallback => 1;
=head1 NAME

    AI::MXNet::Context - A device context.
=cut

=head1 DESCRIPTION

    This class governs the device context of AI::MXNet::NDArray objects.
=cut

=head2

    Constructing a context.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'} or Context.
        String representing the device type

    device_id : int (default=0)
        The device id of the device, needed for GPU
=cut

=head2 cpu

    Returns a CPU context.

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. device_id is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    context : AI::MXNet::Context
        The corresponding CPU context.
=cut

method cpu(Int $device_id=0)
{
    return $self->new(device_type => 'cpu', device_id => $device_id);
}

=head2 gpu

    Returns a GPU context.

    Parameters
    ----------
    device_id : int, optional

    Returns
    -------
    context : AI::MXNet::Context
        The corresponding GPU context.
=cut

method gpu(Int $device_id=0)
{
    return $self->new(device_type => 'gpu', device_id => $device_id);
}

=head2 current_context

    Returns the current context.

    Returns
    -------
    $default_ctx : AI::MXNet::Context
=cut

method current_ctx()
{
    return $AI::MXNet::current_ctx;
}

method set_current(AI::MXNet::Context $current)
{
    $AI::MXNet::current_ctx = $current;
}

*current_context = \&current_ctx;

method deepcopy()
{
    return __PACKAGE__->new(
                device_type => $self->device_type,
                device_id => $self->device_id
    );
}

$AI::MXNet::current_ctx = __PACKAGE__->new(device_type => 'cpu', device_id => 0);

