package AI::MXNet::Symbol::NameManager;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1

    NameManager that does an automatic naming.

    A user can also inherit this object to change the naming behavior.
=cut

has 'counter' => (
    is => 'ro',
    isa => 'HashRef',
    default => sub { +{} }
);

our $current;

=head2 get

    Get the canonical name for a symbol.

    This is default implementation.
    When user specified a name,
    the user specified name will be used.

    When user did not, we will automatically generate a
    name based on hint string.

    Parameters
    ----------
    name : str or undef
        The name the user has specified.

    hint : str
        A hint string, which can be used to generate name.

    Returns
    -------
    full_name : str
        A canonical name for the symbol.
=cut

method get(Maybe[Str] $name, Str $hint)
{
    return $name if $name;
    if(not exists $self->counter->{ $hint })
    {
        $self->counter->{ $hint } = 0;
    }
    $name = sprintf("%s%d", $hint, $self->counter->{ $hint });
    $self->counter->{ $hint }++;
    return $name;
}

method current()
{
    $AI::MXNet::current_nm_ldr;
}

$AI::MXNet::current_nm_ldr = __PACKAGE__->new;

package AI::MXNet::Symbol::Prefix;
use Mouse;

=head1 NAME

    AI::MXNet::Symbol::Prefix
=cut

extends 'AI::MXNet::Symbol::NameManager';

=head1 DESCRIPTION

    A name manager that always attaches a prefix to all names.
=cut

has prefix => (
    is => 'ro',
    isa => 'Str',
    required => 1
);

method get(Maybe[Str] $name, Str $hint)
{
    $name = $self->SUPER::get($name, $hint);
    return $self->prefix . $name;
}

1;
