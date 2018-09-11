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

package AI::MXNet::Engine;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;

=head1 NAME

    AI::MXNet::Engine - Allows management of properties of the MXNet's engine.
=cut

=head1 SYNOPSIS

    my $x;
    mx->engine->bulk(10, sub {
        $x = mx->nd->ones([10]);
        $x *= 2;
        $x += 1;
        $x->wait_to_read();
        $x += 1;
        ok(($x->aspdl == 4)->all);
        for my $i (1..100)
        {
            $x += 1;
        }
    });
    ok(($x->aspdl == 104)->all);
=cut

=head2 set_bulk_size

    Set size limit on bulk execution.

    Bulk execution bundles many operators to run together.
    This can improve performance when running a lot of small
    operators sequentially.

    Parameters
    ----------
    $size : int
        Maximum number of operators that can be bundled in a bulk.

    Returns
    -------
    int
        Previous bulk size.
=cut

method set_bulk_size(Int $size)
{
    return scalar(check_call(AI::MXNetCAPI::EngineSetBulkSize($size)));
}


=head2 bulk

    Bulk execution bundles many operators to run together.
    This can improve performance when running a lot of small
    operators sequentially.

    Parameters
    ----------
    $size : int
        Maximum number of operators that can be bundled in a bulk.
    $sub: CodeRef to execute

    my $x;
    mx->engine->bulk(10, sub {
        $x = mx->nd->zeros([1]);
        for my $i (1..100)
        {
            $x += 1;
        }
    });
=cut

method bulk(Int $size, CodeRef $sub)
{
    my $prev = __PACKAGE__->set_bulk_size($size);
    eval { $sub->() };
    my $err = $@;
    __PACKAGE__->set_bulk_size($prev) unless $prev == $size;
    Carp::confess($err) if $err;
}

1;