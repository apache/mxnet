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
