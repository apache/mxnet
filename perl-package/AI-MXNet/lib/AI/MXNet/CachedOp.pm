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
