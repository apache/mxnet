package AI::MXNet::Symbol::NameManager;
use strict;
use warnings;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1

    NameManager to do automatic naming.

    User can also inherit this object to change naming behavior.
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
        name : str or None
            The name user specified.

        hint : str
            A hint string, which can be used to generate name.

        Returns
        -------
        full_name : str
            A canonical name for the user.
=cut

method get(Str|Undef $name, Str $hint)
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

extends 'AI::MXNet::Symbol::NameManager';

=head1

    A name manager that always attach a prefix to all names.

    Examples
    --------
    >>> import mxnet as mx
    >>> data = mx.symbol.Variable('data')
    >>> with mx.name.Prefix('mynet_'):
            net = mx.symbol.FullyConnected(data, num_hidden=10, name='fc1')
    >>> net.list_arguments()
    ['data', 'mynet_fc1_weight', 'mynet_fc1_bias']
=cut

has prefix => (
    is => 'ro',
    isa => 'Str',
    required => 1
);

method get(Str $name, Str $hint)
{
    $name = $self->SUPER::get($name, $hint);
    return $self->prefix . $name;
}

1;
