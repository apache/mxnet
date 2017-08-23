package AI::MXNet::Gluon::RNN;
use strict;
use warnings;
use AI::MXNet::Gluon::RNN::Layer;
use AI::MXNet::Gluon::RNN::Cell;

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
            \@${short_name}::ISA = ('AI::MXNet::Gluon::RNN_');;
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;