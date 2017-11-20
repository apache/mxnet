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

package AI::MXNet::Random;
use strict;
use warnings;
use Scalar::Util qw/blessed/;
use AI::MXNet::Base;
use AI::MXNet::NDArray::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Random - Handling of randomization in MXNet.
=cut

=head1 DESCRIPTION

    Handling of randomization in MXNet.
=cut

=head2 seed

    Seed the random number generators in mxnet.

    This seed will affect behavior of functions in this module,
    as well as results from executors that contains Random number
    such as Dropout operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    The random number generator of mxnet is by default device specific.
    This means if you set the same seed, the random number sequence
    generated from GPU0 can be different from CPU.
=cut

method seed(Int $seed_state)
{
    check_call(AI::MXNetCAPI::RandomSeed($seed_state));
}

sub AUTOLOAD {
    my $sub = $AI::MXNet::Random::AUTOLOAD;
    $sub =~ s/.*:://;
    shift;
    my %updated;
    my %defaults = (
        ctx   => AI::MXNet::Context->current_ctx,
        shape => 1,
        out   => 1
    );
    my @args;
    my @tmp = @_;
    if(ref $tmp[-1] eq 'HASH')
    {
        my @kwargs = %{ pop(@tmp) };
        push @tmp, @kwargs;
    }
    while(@tmp >= 2 and not ref $tmp[-2])
    {
        if(exists $defaults{$tmp[-2]})
        {
            my $v = pop(@tmp);
            my $k = pop(@tmp);
            if(defined $v)
            {
                $updated{$k} = 1;
                $defaults{$k} = $v;
            }
        }
        else
        {
            unshift @args, pop(@tmp);
            unshift @args, pop(@tmp);
        }
    }
    unshift @args, @tmp;
    if(blessed($defaults{out}) and not exists $updated{shape})
    {
        delete $defaults{shape};
    }
    delete $defaults{out} unless blessed $defaults{out};
    if($sub eq 'exponential')
    {
        my $changed = 0;
        for my $i (0..@args-1)
        {
            if(not ref $args[$i] and $args[$i] eq 'scale')
            {
                $args[$i] = 'lam';
                $args[$i+1] = 1/$args[$i+1];
                $changed = 1;
            }
        }
        $args[0] = 1/$args[0] unless $changed;
    }
    if(grep { blessed($_) and $_->isa('AI::MXNet::NDArray') } @args)
    {
        if($sub eq 'normal')
        {
            my %mapping = qw/loc mu scale sigma/;
            @args = map { (not ref $_ and exists $mapping{$_}) ? $mapping{$_} : $_ } @args
        }
        $sub = "_sample_$sub";
        delete $defaults{shape} if not exists $updated{shape};
        delete $defaults{ctx};
        return AI::MXNet::NDArray->$sub(@args, %defaults);
    }
    else
    {
        $sub = "_random_$sub";
    }
    return AI::MXNet::NDArray->$sub(@args, %defaults);
}

1;
