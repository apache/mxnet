package AI::MXNet::TestUtils;
use strict;
use warnings;
use PDL;
use AI::MXNet::Function::Parameters;
use Exporter;
use base qw(Exporter);
@AI::MXNet::TestUtils::EXPORT_OK = qw(same reldiff almost_equal GetMNIST_ubyte GetCifar10);
use constant default_numerical_threshold => 1e-6;
=head2 same

Test if two numpy arrays are the same

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func same(PDL $a, PDL $b)
{
    return not which($a - $b)->shape->at(0);
}

=head2 reldiff
    Calculate the relative difference between two input arrays

    Calculated by :math:`\\frac{|a-b|_1}{|a|_1 + |b|_1}`

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func reldiff(PDL $a, PDL $b)
{
    my $diff = sum(abs($a - $b));
    my $norm = sum(abs($a)) + sum(abs($b));
    if($diff == 0)
    {
        return 0;
    }
    my $ret = $diff / $norm;
    return $ret;
}

=head2 almost_equal

Test if two pdl arrays are almost equal.
=cut

func almost_equal(PDL $a, PDL $b, Maybe[Num] $threshold=)
{
    $threshold //= default_numerical_threshold;
    my $rel = reldiff($a, $b);
    return $rel <= $threshold;
}

func GetMNIST_ubyte()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (
        not -f 'data/train-images-idx3-ubyte'
            or
        not -f 'data/train-labels-idx1-ubyte'
            or
        not -f 'data/t10k-images-idx3-ubyte'
            or
        not -f 'data/t10k-labels-idx1-ubyte'
    )
    {
        `wget http://data.mxnet.io/mxnet/data/mnist.zip -P data`;
        chdir 'data';
        `unzip -u mnist.zip`;
        chdir '..';
    }
}

func GetCifar10()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (not -f 'data/cifar10.zip')
    {
        `wget http://data.mxnet.io/mxnet/data/cifar10.zip -P data`;
        chdir 'data';
        `unzip -u cifar10.zip`;
        chdir '..';
    }
}


1;