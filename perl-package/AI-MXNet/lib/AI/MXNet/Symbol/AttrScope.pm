package AI::MXNet::Symbol::AttrScope;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 

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
    default => sub { +{} }
);

=head2 current

        Get the attribute dict given the attribute set by the symbol.

        Parameters
        ----------
        None.

        Returns
        -------
        attr : current value of the class singleton object
=cut

method current()
{
    $AI::MXNet::curr_attr_scope; 
}

=head2 get

        Get the attribute dict given the attribute set by the symbol.

        Parameters
        ----------
        attr : dict of string to string
            The attribute passed in by user during symbol creation.

        Returns
        -------
        attr : dict of string to string
            Updated attributes to add other scope related attributes.
=cut
 
method get(HashRef[Str]|Undef $attr=)
{
    return bless($attr//{}, 'AI::MXNet::Util::Printable') unless %{ $self->attr }; 
    my %ret = (%{ $self->attr }, %{ $attr//{} });
    return bless (\%ret, 'AI::MXNet::Util::Printable');
}

$AI::MXNet::curr_attr_scope = __PACKAGE__->new;
