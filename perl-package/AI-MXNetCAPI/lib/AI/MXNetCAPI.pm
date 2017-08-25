package AI::MXNetCAPI;
use base qw(DynaLoader);
bootstrap AI::MXNetCAPI;
our $VERSION = '1.01';
1;
__END__

=head1 NAME

AI::MXNetCAPI - Swig interface to mxnet c api

=head1 SYNOPSIS

 use AI::MXNetCAPI;

=head1 DESCRIPTION

This module provides interface to mxnet
via its api.

=head1 SEE ALSO

L<AI::MXNet>

=head1 AUTHOR

Sergey Kolychev, <sergeykolychev.github@gmail.com>

=head1 COPYRIGHT & LICENSE

Copyright 2017 Sergey Kolychev.

This library is licensed under Apache 2.0 license.

See https://www.apache.org/licenses/LICENSE-2.0 for more information.

=cut
