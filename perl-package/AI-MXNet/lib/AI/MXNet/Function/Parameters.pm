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

package AI::MXNet::Function::Parameters;
use strict;
use warnings;
use Function::Parameters ();
use AI::MXNet::Types ();
sub import {
    Function::Parameters->import(
        {
            func => {
                defaults => 'function_strict',
                runtime  => 1,
                reify_type => sub {
                    Mouse::Util::TypeConstraints::find_or_create_isa_type_constraint($_[0])
                }
            },
            method => {
                defaults => 'method_strict',
                runtime  => 1,
                reify_type => sub {
                    Mouse::Util::TypeConstraints::find_or_create_isa_type_constraint($_[0])
                }
            },
        }
    );
}

{
    no warnings 'redefine';
    *Function::Parameters::_croak = sub {
        local($Carp::CarpLevel) = 1;
        Carp::confess ("@_");
    };
}

1;
