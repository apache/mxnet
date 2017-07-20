package AI::MXNet::Logging;
## TODO
use Mouse;
sub warning { shift; warn sprintf(shift, @_) . "\n" };
*debug   = *info = *warning;
sub get_logger { __PACKAGE__->new }

1;
