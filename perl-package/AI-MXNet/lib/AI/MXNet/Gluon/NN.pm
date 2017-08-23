package AI::MXNet::Gluon::NN;
use strict;
use warnings;
use AI::MXNet::Gluon::Block;
use AI::MXNet::Gluon::NN::BasicLayers;
use AI::MXNet::Gluon::NN::ConvLayers;

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
            \@${short_name}::ISA = ('AI::MXNet::Gluon::NN_');
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;