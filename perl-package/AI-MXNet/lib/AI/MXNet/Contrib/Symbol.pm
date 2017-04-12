package AI::MXNet::Contrib::Symbol;
use strict;
use warnings;

sub AUTOLOAD {
    my $sub = $AI::MXNet::Contrib::Symbol::AUTOLOAD;
    $sub =~ s/.*:://;
    $sub = "_contrib_$sub";
    shift;
    return AI::MXNet::Symbol->$sub(@_);
}

1;