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

package AI::MXNet::Gluon;
use strict;
use warnings;
use AI::MXNet::Gluon::Loss;
use AI::MXNet::Gluon::Trainer;
use AI::MXNet::Gluon::Utils;
use AI::MXNet::Gluon::Data;
use AI::MXNet::Gluon::NN;
use AI::MXNet::Gluon::RNN;

sub import
{
    my ($class, $short_name) = @_;
    if($short_name)
    {
        $short_name =~ s/[^\w:]//g;
        if(length $short_name)
        {
            my $short_name_package =<<"EOP";
            package $short_name;
            sub data { 'AI::MXNet::Gluon::Data' }
            sub nn { 'AI::MXNet::Gluon::NN_' }
            sub rnn { 'AI::MXNet::Gluon::RNN_' }
            sub loss { 'AI::MXNet::Gluon::Loss_' }
            sub utils { 'AI::MXNet::Gluon::Utils' }
            sub Trainer { shift; AI::MXNet::Gluon::Trainer->new(\@_); }
            sub Parameter { shift; AI::MXNet::Gluon::Parameter->new(\@_); }
            sub ParameterDict { shift; AI::MXNet::Gluon::ParameterDict->new(\@_); }
            \@${short_name}::ISA = ('AI::MXNet::Gluon_');
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;