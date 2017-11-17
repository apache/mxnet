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

package AI::MXNet::Monitor;
use Mouse;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;

=head1 NAME

    AI::MXNet::Monitor - Monitor outputs, weights, and gradients for debugging.

=head1 DESCRIPTION

    Monitor outputs, weights, and gradients for debugging.

    Parameters
    ----------
    interval : int
        Number of batches between printing.
    stat_func : function
        a function that computes statistics of tensors.
        Takes a NDArray and returns a NDArray. defaults to mean
        absolute value |x|/size(x).
    pattern : str
        A regular expression specifying which tensors to monitor.
        Only tensors with names that match name_pattern will be included.
        For example, '.*weight|.*output' will print all weights and outputs;
        '.*backward.*' will print all gradients.
=cut

has 'interval'  => (is => 'ro', isa => 'Int', required => 1);
has 'stat_func' => (
    is => 'ro',
    isa => 'CodeRef',
    default => sub {
        return sub {
            # returns |x|/size(x), async execution.
            my ($x) = @_;
            return $x->norm/sqrt($x->size);
        }
    },
    lazy => 1
);
has 'pattern'             => (is => 'ro', isa => 'Str', default => '.*');
has '_sort'               => (is => 'ro', isa => 'Bool', init_arg => 'sort', default => 0);
has [qw/queue exes/]      => (is => 'rw', init_arg => undef, default => sub { [] });
has [qw/step activated/]  => (is => 'rw', init_arg => undef, default => 0);
has 're_pattern'          => (
    is => 'ro',
    init_arg => undef,
    default => sub {
        my $pattern = shift->pattern;
        my $re = eval { qr/$pattern/ };
        confess("pattern $pattern failed to compile as a regexp $@")
            if $@;
        return $re;
    },
    lazy => 1
);
has 'stat_helper'          => (
    is => 'ro',
    init_arg => undef,
    default => sub {
        my $self = shift;
        return sub {
            my ($name, $handle) = @_;
            return if(not $self->activated or not $name =~ $self->re_pattern);
            my $array = AI::MXNet::NDArray->new(handle => $handle, writable => 0);
            push @{ $self->queue }, [$self->step, $name, $self->stat_func->($array)];
        }
    },
    lazy => 1
);

=head2 install

    install callback to executor.
    Supports installing to multiple exes.

    Parameters
    ----------
    exe : AI::MXNet::Executor
        the Executor (returned by $symbol->bind) to install to.
=cut

method install(AI::MXNet::Executor $exe)
{
    $exe->set_monitor_callback($self->stat_helper);
    push @{ $self->exes }, $exe;
}

=head2 tic

    start collecting stats for current batch.
    Call before forward
=cut

method tic()
{
        if ($self->step % $self->interval == 0)
        {
            for my $exe (@{ $self->exes })
            {
                $_->wait_to_read for @{ $exe->arg_arrays };
                $_->wait_to_read for @{ $exe->aux_arrays };
            }
            $self->queue([]);
            $self->activated(1);
        }
        $self->step($self->step + 1);
}

=head2 toc

    End collecting for current batch and return results.
    Call after computation of current batch.

    Returns
    -------
    res : array ref of array refs with debug info
=cut

method toc()
{
    return [] unless $self->activated;
    for my $exe (@{ $self->exes })
    {
        $_->wait_to_read for @{ $exe->arg_arrays };
        $_->wait_to_read for @{ $exe->aux_arrays };
    }
    for my $exe (@{ $self->exes })
    {
        for(zip($exe->_symbol->list_arguments, $exe->arg_arrays)) {
            my ($name, $array) = @$_;
            push @{ $self->queue }, [$self->step, $name, $self->stat_func->($array)];
        }
        for(zip($exe->_symbol->list_auxiliary_states, $exe->aux_arrays)) {
            my ($name, $array) = @$_;
            push @{ $self->queue }, [$self->step, $name, $self->stat_func->($array)];
        }
    }
    $self->activated(0);
    my @res;
    if($self->_sort)
    {
        @{ $self->queue } = sort { $a->[1] cmp $b->[1] } @{ $self->queue };
    }
    for my $q (@{ $self->queue })
    {
        my ($n, $k, $v_list) = @{ $q };
        if(ref $v_list ne 'ARRAY')
        {
            $v_list = [$v_list];
        }
        my $s = '';
        for my $v (@{ $v_list })
        {
            confess("the argument must be NDArray")
                unless blessed($v) and $v->isa('AI::MXNet::NDArray');
            if($v->size == 1)
            {
                $s .= $v->asscalar . "\t";
            }
            else
            {
                $s .= $v->aspdl . "\t";
            }
        }
        push @res, [$n, $k, $s];
    }
    $self->queue([]);
    return \@res;
}

=head2 toc_print

    End collecting and print results
=cut

method toc_print()
{
    my $res = $self->toc;
    for my $r (@{ $res })
    {
        AI::MXNet::Logging->info('Batch: %7d %30s %s', @{ $r });
    }
}

method Monitor(@args)
{
    __PACKAGE__->new(@args % 2 ? ('interval', @args) : @args);
}

1;
