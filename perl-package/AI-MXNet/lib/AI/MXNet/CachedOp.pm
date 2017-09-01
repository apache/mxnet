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

package AI::MXNet::CachedOp;

=head1 NAME

    AI::MXNet::CachedOp - A wrapper around CachedOpHandle
=cut

use strict;
use warnings;
use AI::MXNet::Base;
use Mouse;
use overload '&{}' => sub { my $self = shift; sub { $self->call(@_) } };

has 'handle'   => (is => 'ro', isa => 'CachedOpHandle', required => 1);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    my ($sym) = @_;
    my $handle = check_call(
        AI::MXNetCAPI::CreateCachedOp(
            $sym->handle
        )
    );
    return $class->$orig(handle => $handle);
};

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::FreeCachedOp(shift->handle));
}

sub call
{
    my $self = shift;
    my @args;
    my %kwargs;
    if(blessed $_[0] and $_[0]->isa('AI::MXNet::NDArray'))
    {
        while(blessed $_[0] and $_[0]->isa('AI::MXNet::NDArray'))
        {
            push @args, shift(@_);
        }
        %kwargs = @_;
    }
    else
    {
        %kwargs = @_;
    }
    my $out = delete $kwargs{out};
    if(%kwargs)
    {
        confess(
            "AI::MXNet::CachedOp::call got unexpected keyword argument(s): ".
            join(', ', keys %kwargs)
        );
    }
    my $original_output;
    if(defined $out)
    {
        $original_output = $out;
        if(blessed($out))
        {
            $out = [$out];
        }
    }
    else
    {
        $out = [];
    }
    my $output = check_call(
        AI::MXNetCAPI::InvokeCachedOp(
            $self->handle,
            scalar(@args),
            [map { $_->handle } @args],
            [map { $_->handle } @$out]
        )
    );
    return $original_output if defined $original_output;
    if(@$output == 1)
    {
        return AI::MXNet::NDArray->new(handle => $output->[0]);
    }
    else
    {
        return [map { AI::MXNet::NDArray->new(handle => $_) } @$output];
    }
}

1;
