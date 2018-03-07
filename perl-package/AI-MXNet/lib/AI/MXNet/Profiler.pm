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

package AI::MXNet::Profiler;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Profiler - Optional profiler feature.
=cut

=head1 DESCRIPTION

    Optional profirer.
=cut

=head2 profiler_set_config

    Set up the configure of profiler.

    Parameters
    ----------
    kwargs : hash ref
        Indicates configuration parameters with key/value pairs, listed below
          profile_symbolic : boolean, whether to profile symbolic operators
          profile_imperative : boolean, whether to profile imperative operators
          profile_memory : boolean, whether to profile memory usage
          profile_api : boolean, whether to profile the C API
          file_name : string, output file for profile data
          continuous_dump : boolean, whether to periodically dump profiling data to file
          dump_period : float, seconds between profile data dumps
=cut

method profiler_set_config(HashRef[Str] $kwargs)
{
    check_call(AI::MXNet::SetProfilerConfig(scalar(keys %{ $kwargs }), $kwargs));
}

=head2 profiler_set_state

    Set up the profiler state to record operator.

    Parameters
    ----------
    state : int, optional
        Indicting whether to run the profiler, can
        be 'stop' - 0 or 'run' - 1. Default is `stop`.
=cut

method profiler_set_state(Int $state)
{
    check_call(AI::MXNet::SetProfilerState($state));
}

=head2 dump_profile

    Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally
=cut

method dump_profile()
{
    check_call(AI::MXNetCAPI::DumpProfile());
}

1;
