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

package AI::MXNet::Gluon::Mouse;
use strict;
use warnings;
use Mouse;
use Mouse::Exporter;
no Mouse;

Mouse::Exporter->setup_import_methods(
    as_is   => [
        'has',
        \&Mouse::extends,
        \&Mouse::with,
        \&Mouse::before,
        \&Mouse::after,
        \&Mouse::around,
        \&Mouse::override,
        \&Mouse::super,
        \&Mouse::augment,
        \&Mouse::inner,
        \&Scalar::Util::blessed,
        \&Carp::confess
    ]
);

sub init_meta { return Mouse::init_meta(@_) }
sub has
{
    my $name = shift;
    my %args = @_;
    my $caller = delete $args{caller} // caller;
    my $meta = $caller->meta;

    $meta->throw_error(q{Usage: has 'name' => ( key => value, ... )})
        if @_ % 2; # odd number of arguments

    for my $n (ref($name) ? @{$name} : $name){
        $meta->add_attribute(
            $n,
            trigger => sub { my $self = shift; $self->__setattr__($n, @_); },
            %args
        );
    }
    return;
}

1;