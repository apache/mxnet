package AI::MXNet::Logging;
## TODO
use strict;
use warnings;
use Mouse;
our $silent = 0;
sub warning { return if $silent; shift; warn sprintf(shift, @_) . "\n" };
*debug   = *info = *warning;
sub get_logger { __PACKAGE__->new }

1;
