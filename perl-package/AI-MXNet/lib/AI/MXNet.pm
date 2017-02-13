package AI::MXNet;
use v5.14.0;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Callback;
use AI::MXNet::NDArray;
use AI::MXNet::Symbol;
use AI::MXNet::Executor;
use AI::MXNet::Executor::Group;
use AI::MXNet::Rtc;
use AI::MXNet::Random;
use AI::MXNet::Initializer;
use AI::MXNet::Optimizer;
use AI::MXNet::KVStore;
use AI::MXNet::KVStoreServer;
use AI::MXNet::IO;
use AI::MXNet::Metric;
use AI::MXNet::LRScheduler;
use AI::MXNet::Monitor;
use AI::MXNet::Profiler;
use AI::MXNet::Module::Base;
use AI::MXNet::Module;
our $VERSION = '0.01';

sub import
{
    my ($class, $short_name) = @_;
    if($short_name)
    {
        $short_name =~ s/[^\w:]//g;
        if(length $short_name)
        {
            my $short_name_package =<<"EOP";
            package $short_name;
            sub nd { 'AI::MXNet::NDArray' }
            sub sym { 'AI::MXNet::Symbol' }
            sub symbol { 'AI::MXNet::Symbol' }
            sub init { 'AI::MXNet::Initializer' }
            sub initializer { 'AI::MXNet::Initializer' }
            sub optimizer { 'AI::MXNet::Optimizer' }
            sub opt { 'AI::MXNet::Optimizer' }
            sub rnd { 'AI::MXNet::Random' }
            sub random { 'AI::MXNet::Random' }
            sub cpu { AI::MXNet::Context->cpu(\$_[1]//0) }
            sub gpu { AI::MXNet::Context->gpu(\$_[1]//0) }
            sub kv { 'AI::MXNet::KVStore' }
            sub io { 'AI::MXNet::IO' }
            sub metric { 'AI::MXNet::Metric' }
            sub mod { 'AI::MXNet::Module' }
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;
__END__

=encoding UTF-8

=head1 NAME

AI::MXNet - Perl interface to MXNet machine learning library

=head1 SYNOPSIS

    This is alpha release.
    Please refer to t dir for examples.

=head1 DESCRIPTION

    Perl interface to MXNet machine learning library.

=head1 BUGS AND INCOMPATIBILITIES

Parity with Python inteface is not yet achieved.
Pod mostly contains Python documentation taken as is.
This is WIP.

=head1 SEE ALSO

http://mxnet.io/

=head1 AUTHOR

Sergey Kolychev, <sergeykolychev.github@gmail.com>

=head1 COPYRIGHT & LICENSE

Copyright 2017 Sergey Kolychev.

This program is free software; you can redistribute it and/or modify it
under the terms of either: the GNU General Public License as published
by the Free Software Foundation; or the Artistic License.

See http://dev.perl.org/licenses/ for more information.

=cut
