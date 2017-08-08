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

package AI::MXNet::Symbol::AttrScope;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Function::Parameters;
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(attr => {@_});
};

=head1 NAME

    AI::MXNet::Symbol::AttrScope - Attribute manager for local scoping.

=head1 DESCRIPTION

    Attribute manager for scoping.

    User can also inherit this object to change naming behavior.

    Parameters
    ----------
    kwargs
        The attributes to set for all symbol creations in the scope.
=cut

has 'attr' => (
    is => 'ro',
    isa => 'HashRef[Str]',
);

=head2 current

    Get the attribute hash ref given the attribute set by the symbol.

    Returns
    -------
    $attr : current value of the class singleton object
=cut

method current()
{
    $AI::MXNet::curr_attr_scope;
}

=head2 get

    Get the attribute hash ref given the attribute set by the symbol.

    Parameters
    ----------
    $attr : Maybe[HashRef[Str]]
        The attribute passed in by user during symbol creation.

    Returns
    -------
    $attr : HashRef[Str]
        The attributes updated to include another the scope related attributes.
=cut

method get(Maybe[HashRef[Str]] $attr=)
{
    return bless($attr//{}, 'AI::MXNet::Util::Printable') unless %{ $self->attr };
    my %ret = (%{ $self->attr }, %{ $attr//{} });
    return bless (\%ret, 'AI::MXNet::Util::Printable');
}

$AI::MXNet::curr_attr_scope = __PACKAGE__->new;
