package AI::MXNet::Contrib;
use strict;
use warnings;
use AI::MXNet::Contrib::Symbol;
use AI::MXNet::Contrib::NDArray;

sub sym    { 'AI::MXNet::Contrib::Symbol'  }
sub symbol { 'AI::MXNet::Contrib::Symbol'  }
sub nd     { 'AI::MXNet::Contrib::NDArray' }
sub autograd { 'AI::MXNet::Contrib::AutoGrad' }

1;