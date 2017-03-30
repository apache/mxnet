package AI::MXNet::Contrib::NDArray;
use strict;
use warnings;

sub AUTOLOAD {
    my $sub = $AI::MXNet::Contrib::NDArray::AUTOLOAD;
    $sub =~ s/.*:://;
    $sub = "_contrib_$sub";
    shift;
    return AI::MXNet::NDArray->$sub(@_);
}

1;