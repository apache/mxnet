package AI::MXNet::Util::Printable;
use strict;
use warnings;
use Data::Dumper qw();
use overload '""' => sub { print Data::Dumper->new([shift])->Purity(1)->Deepcopy(1)->Terse(1)->Dump };