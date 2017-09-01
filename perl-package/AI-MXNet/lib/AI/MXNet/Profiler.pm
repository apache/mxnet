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
    mode : string, optional
        Indicting whether to enable the profiler, can
        be 'symbolic' or 'all'. Default is `symbolic`.
    filename : string, optional
        The name of output trace file. Default is
        'profile.json'.
=cut

method profiler_set_config(ProfilerMode $mode='symbolic', Str $filename='profile.json')
{
    my %mode2int = qw/symbolic 0 all 1/;
    check_call(AI::MXNet::SetProfilerConfig($mode2int{ $mode }, $filename));
}

=head2 profiler_set_state

    Set up the profiler state to record operator.

    Parameters
    ----------
    state : string, optional
        Indicting whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
=cut

method profiler_set_state(ProfilerState $state='stop')
{
    my %state2int = qw/stop 0 run 1/;
    check_call(AI::MXNet::SetProfilerState($state2int{ $state }));
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
