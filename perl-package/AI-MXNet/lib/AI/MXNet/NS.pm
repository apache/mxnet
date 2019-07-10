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

package AI::MXNet::NS;
# this class is similar to Exporter, in that it will add an "import"
# method to the calling package.  It is to allow a package to emulate
# the python "import mxnet as mx" style aliasing as "use AI::MXNet 'mx'"
use strict;
use warnings;

sub _sym : lvalue
{
    my ($pkg, $name) = @_;
    no strict 'refs';
    *{"$pkg\::$name"};
}

sub import
{
    my (undef, $opt) = @_;
    my $class = caller();
    my $func = sub { $class };
    _sym($class, 'import') = sub {
        my (undef, @names) = @_;
        @names = map { s/[^\w:]//sgr } @names;
        my $target = caller();

        _sym($names[0], '') = _sym($class, '') if
            @names == 1 and $opt and $opt eq 'global';

        _sym($target, $_) = $func for @names;
    };
}

my $autoload_template = q(
    sub AUTOLOAD
    {
        our ($AUTOLOAD, %AUTOLOAD);
        my $name = $AUTOLOAD =~ s/.*:://sr;
        my $func = $AUTOLOAD{$name};
        Carp::carp(qq(Can't locate object method "$name" via package "${\ __PACKAGE__ }"))
            unless $func;
        goto $func;
    }
);

# using AUTOLOAD here allows for the addition of an AI::MXNet::SomeClass
# class to coexist with an AI::MXNet->SomeClass() shorthand constructor.
sub register
{
    my ($class, $target) = @_;
    my $name = $class =~ s/.*:://sr;
    my $dest = $class->can('new');
    ${_sym($target, 'AUTOLOAD')}{$name} = sub {
        splice @_, 0, 1, $class;
        goto $dest;
    };
    return if $target->can('AUTOLOAD');
    eval sprintf 'package %s { %s }', $target, $autoload_template;
    die if $@;
    return;
}

1;
