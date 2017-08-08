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

for my $method (
        [qw/_sample_uniform uniform/],
        [qw/_sample_normal normal/],
        [qw/_sample_gamma gamma/],
        [qw/_sample_exponential exponential/],
        [qw/_sample_poisson poisson/],
        [qw/_sample_negbinomial negative_binomial/],
        [qw/_sample_gennegbinomial generalized_negative_binomial/],
)
{
    my ($nd_method_name, $rnd_method_name) = @{$method};
    {
        no strict 'refs';
        *{__PACKAGE__."::$rnd_method_name"} = sub { shift;
            return AI::MXNet::NDArray->$nd_method_name(@_);
        };
    }
}

1;
