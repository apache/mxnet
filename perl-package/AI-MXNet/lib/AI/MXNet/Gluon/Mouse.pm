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