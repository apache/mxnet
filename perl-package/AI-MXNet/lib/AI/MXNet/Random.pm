package AI::MXNet::Random;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::Random - Handling of randomization in MXNet.
=cut

=head1 DESCRIPTION

    Handling of randomization in MXNet.
=cut

=head2 seed

    Seed the random number generators in mxnet.

    This seed will affect behavior of functions in this module,
    as well as results from executors that contains Random number
    such as Dropout operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    The random number generator of mxnet is by default device specific.
    This means if you set the same seed, the random number sequence
    generated from GPU0 can be different from CPU.
=cut

method seed(Int $seed_state)
{
    check_call(AI::MXNetCAPI::RandomSeed($seed_state));
}

for my $method (
        [qw/_sample_uniform uniform/],
        [qw/_sample_normal normal/],
        [qw/_sample_gamma gamma/],
        [qw/_sample_exponential exponential/],
        [qw/_sample_poisson poisson/],
        [qw/_sample_negbinomial negative_binomial/],
        [qw/_sample_gennegbinomial generalized_negative_binomial/],
)
{
    my ($nd_method_name, $rnd_method_name) = @{$method};
    {
        no strict 'refs';
        *{__PACKAGE__."::$rnd_method_name"} = sub { shift;
            return AI::MXNet::NDArray->$nd_method_name(@_);
        };
    }
}

1;