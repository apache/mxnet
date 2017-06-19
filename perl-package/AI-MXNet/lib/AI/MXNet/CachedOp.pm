package AI::MXNet::CachedOp;

=head1 NAME

    AI::MXNet::CachedOp - A wrapper around CachedOpHandle
=cut

use strict;
use warnings;
use AI::MXNet::Base;
use Mouse;

has 'op'       => (is => 'ro', isa => 'Str', required => 1);
has 'handle'   => (is => 'ro', isa => 'CachedOpHandle', required => 1);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    my ($op, $num_input, %kwargs) = @_;
    for my $key (keys %kwargs)
    {
        $kwargs{ $key } = "(" .join(", ", @{ $kwargs{ $key } }) .")"
            if ref $kwargs{ $key } eq 'ARRAY';
    }
    my $AtomicSymbolCreator = check_call(AI::NNVMCAPI::GetOpHandle($op));
    my $handle = check_call(
        AI::MXNetCAPI::CachedCreateOp(
            $AtomicSymbolCreator,
            $num_input,
            scalar(keys %kwargs),
            \%kwargs
        )
    );
    return $class->$orig(op => $op, handle => $handle);
};

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::CachedFree(shift->handle));
}

1;