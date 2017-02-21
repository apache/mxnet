# Base class for Initializer
package AI::MXNet::Initializer;
use strict;
use warnings;
use AI::MXNet::Base qw(:DEFAULT pzeros pceil);
use AI::MXNet::NDArray;
use Mouse;
use AI::MXNet::Function::Parameters;
for my $class (qw/Mixed Uniform Normal Orthogonal Xavier MSRAPrelu/)
{
    { no strict 'refs'; *{"$class"} = sub { shift; "AI::MXNet::$class"->new(@_) } }
}
use overload "&{}" => sub { my $self = shift; sub { $self->call(@_) } };

=head2 init

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
=cut

method call(Str $name, AI::MXNet::NDArray $arr)
{
    if($name =~ /^upsampling/)
    {
        $self->_init_bilinear($name, $arr);
    }
    elsif($name =~ /^stn_loc/ and $name =~ /weight$/)
    {
        $self->_init_zero($name, $arr);
    }
    elsif($name =~ /^stn_loc/ and $name =~ /bias$/)
    {
        $self->_init_loc_bias($name, $arr);
    }
    elsif($name =~ /bias$/)
    {
        $self->_init_bias($name, $arr);
    }
    elsif($name =~ /gamma$/)
    {
        $self->_init_gamma($name, $arr);
    }
    elsif($name =~ /beta$/)
    {
        $self->_init_beta($name, $arr);
    }
    elsif($name =~ /weight$/)
    {
        $self->_init_weight($name, $arr);
    }
    elsif($name =~ /moving_mean$/)
    {
        $self->_init_zero($name, $arr);
    }
    elsif($name =~ /moving_var$/)
    {
        $self->_init_one($name, $arr);
    }
    elsif($name =~ /moving_inv_var$/)
    {
        $self->_init_zero($name, $arr);
    }
    elsif($name =~ /moving_avg$/)
    {
        $self->_init_zero($name, $arr);
    }
    else
    {
        $self->_init_default($name, $arr);
    }
}

*slice = *call;

method _init_bilinear($name, $arr)
{
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ 'float32' });
    my $weight = pzeros(
        PDL::Type->new(DTYPE_MX_TO_PDL->{ 'float32' }),
        $arr->size
    );
    my $shape = $arr->shape;
    my $size = $arr->size;
    my $f = pceil($shape->[3] / 2)->at(0);
    my $c = (2 * $f - 1 - $f % 2) / (2 * $f);
    for my $i (0..($size-1))
    {
        my $x = $i % $shape->[3];
        my $y = ($i / $shape->[3]) % $shape->[2];
        $weight->index($i) .= (1 - abs($x / $f - $c)) * (1 - abs($y / $f - $c));
    }
    $arr .= $weight->reshape(reverse @{ $shape });
}

method _init_loc_bias($name, $arr)
{
    confess("assert error shape[0] == 6")
        unless $arr->shape->[0] == 6;
    $arr .= [1.0, 0, 0, 0, 1.0, 0];
}

method _init_zero($name, $arr)
{
    $arr .= 0;
}

method _init_one($name, $arr)
{
    $arr .= 1;
}

method _init_bias($name, $arr)
{
    $arr .= 0;
}

method _init_gamma($name, $arr)
{
    $arr .= 1;
}

method _init_beta($name, $arr)
{
    $arr .= 0;
}

method _init_weight($name, $arr)
{
    confess("Virtual method, subclass must override it");
}

method _init_default($name, $arr)
{
    confess("Unknown initialization pattern for $name");
}

=head2

    Initialize by loading pretrained param from file or dict

    Parameters
    ----------
    param: str or dict of str->NDArray
        param file or dict mapping name to NDArray.
    default_init: Initializer
        default initializer when name is not found in param.
    verbose: bool
        log source when initializing.
=cut

package AI::MXNet::Load;
use Mouse;
extends 'AI::MXNet::Initializer';

has 'param' => (is => "rw", isa => 'HashRef[AI::MXNet::NDArray]', required => 1);
has 'default_init' => (is => "rw", isa => "AI::MXNet::Initializer");
has 'verbose' => (is => "rw", isa => "Int", default => 0);

sub BUILD
{
    my $self = shift;
    my $param = AI::MXNet::NDArray->load($self->param) unless ref $self->param;
    my %self_param;
    while(my ($name, $arr) = each %{ $self->param })
    {
        $name =~ s/^(?:arg|aux)://;
        $self_param{ $name } = $arr;
    }
    $self->param(\%self_param);
}

method call(Str $name, AI::MXNet::NDArray $arr)
{
    if(exists $self->param->{ $name })
    {
        my $target_shape = join(',', @{ $arr->shape });
        my $param_shape  = join(',', @{ $self->param->{ $name }->shape });
        confess(
            "Parameter $name cannot be initialized from loading. "
            ."Shape mismatch, target $target_shape vs loaded $param_shape"
        ) unless $target_shape eq $param_shape;
        $arr .= $self->param->{ $name };
        AI::MXNet::Log->info("Initialized $name by loading") if $self->verbose;
    }
    else
    {
        confess(
            "Cannot Initialize $name. Not found in loaded param "
            ."and no default Initializer is provided."
        ) unless defined $self->default_init;
        $self->default_init($name, $arr);
        AI::MXNet::Log->info("Initialized $name by default") if $self->verbose;
    }
}

*slice = *call;

=begin
    Initialize with mixed Initializer

    Parameters
    ----------
    patterns: list of str
        list of regular expression patterns to match parameter names.
    initializers: list of Initializer
        list of Initializer corrosponding to patterns
=cut

package AI::MXNet::Mixed;
use Mouse;
extends 'AI::MXNet::Initializer';

has "map"          => (is => "rw", init_arg => undef);
has "patterns"     => (is => "ro", isa => 'ArrayRef[Str]');
has "initializers" => (is => "ro", isa => 'ArrayRef[AI::MXnet::Initializer]');

sub BUILD
{
    my $self = shift;
    confess("patterns count != initializers count")
        unless (@{ $self->patterns } == @{ $self->initializers });
    my %map;
    @map{ @{ $self->patterns } } = @{ $self->initializers };
    $self->map(\%map);
}

method call(Str $name, AI::MXNet::NDArray $arr)
{
    for my $pattern (keys %{ $self->map })
    {
        if($name =~ /$pattern/)
        {
            &{$self->map->{$pattern}}($name, $arr);
            return;
        }
    }
    confess(
        "Parameter name $name did not match any pattern. Consider"
        ."add a \".*\" pattern at the and with default Initializer."
    );
}

=begin

    Initialize the weight with uniform [-scale, scale]

    Parameters
    ----------
    scale : float, optional
        The scale of uniform distribution
=cut

package AI::MXNet::Uniform;
use Mouse;
extends 'AI::MXNet::Initializer';

has "scale" => (is => "ro", isa => "Num", default => 0.7);

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    AI::MXNet::Random->uniform(-$self->scale, $self->scale, { out => $arr });
}

=begin

    Initialize the weight with normal(0, sigma)

    Parameters
    ----------
    sigma : float, optional
        Standard deviation for gaussian distribution.
=cut

package AI::MXNet::Normal;
use Mouse;
extends 'AI::MXNet::Initializer';

has "sigma" => (is => "ro", isa => "Num", default => 0.01);

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    AI::MXNet::Random->normal(0, $self->sigma, { out => $arr });
}

=begin
    Intialize weight as Orthogonal matrix

    Parameters
    ----------
    scale : float optional
        scaling factor of weight

    rand_type: string optional
        use "uniform" or "normal" random number to initialize weight

    Reference
    ---------
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    arXiv preprint arXiv:1312.6120 (2013).
=cut

package AI::MXNet::Orthogonal;
use AI::MXNet::Base;
use Mouse;
use AI::MXNet::Types;
extends 'AI::MXNet::Initializer';

has "scale" => (is => "ro", isa => "Num", default => 1.414);
has "randtype" => (is => "ro", isa => enum([qw/uniform normal/]), default => 'uniform');

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    my @shape = @{ $arr->shape };
    my $nout = $shape[0];
    my $nin = AI::MXNet::NDArray->size([@shape[1..$#shape]]);
    my $tmp = AI::MXNet::NDArray->zeros([$nout, $nin]);
    if($self->randtype eq 'uniform')
    {
        AI::MXNet::Random->uniform(-1, 1, { out => $tmp });
    }
    else
    {
        AI::MXNet::Random->normal(0, 1, { out => $tmp });
    }
    $tmp = $tmp->aspdl;
    my ($u, $s, $v) = svd($tmp);
    my $q;
    if(join(',', @{ $u->shape->unpdl }) eq join(',', @{ $tmp->shape->unpdl }))
    {
        $q = $u;
    }
    else
    {
        $q = $v;
    }
    $q = $self->scale * $q->reshape(reverse(@shape));
    $arr .= $q;
}

*slice = *call;

=begin

    Initialize the weight with Xavier or similar initialization scheme.

    Parameters
    ----------
    rnd_type: str, optional
        Use ```gaussian``` or ```uniform``` to init

    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    magnitude: float, optional
        scale of random number range
=cut

package AI::MXNet::Xavier;
use Mouse;
use AI::MXNet::Types;
extends 'AI::MXNet::Initializer';

has "magnitude" => (is => "rw", isa => "Num", default => 3);
has "rnd_type" => (is => "ro", isa => enum([qw/uniform gaussian/]), default => 'uniform');
has "factor_type" => (is => "ro", isa => enum([qw/avg in out/]), default => 'avg');

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    my @shape = @{ $arr->shape };
    my $hw_scale = 1;
    if(@shape > 2)
    {
        $hw_scale = AI::MXNet::NDArray->size([@shape[2..$#shape]]);
    }
    my ($fan_in, $fan_out) = ($shape[1] * $hw_scale, $shape[0] * $hw_scale);
    my $factor;
    if($self->factor_type eq "avg")
    {
        $factor = ($fan_in + $fan_out) / 2;
    }
    elsif($self->factor_type eq "in")
    {
        $factor = $fan_in;
    }
    else
    {
        $factor = $fan_out;
    }
    my $scale = sqrt($self->magnitude / $factor);
    if($self->rnd_type eq "iniform")
    {
        AI::MXNet::Random->uniform(-$scale, $scale, { out => $arr });
    }
    else
    {
        AI::MXNet::Random->normal(0, $scale, { out => $arr });
    }
}

=begin

    Initialize the weight with initialization scheme from
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.

    Parameters
    ----------
    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    slope: float, optional
        initial slope of any PReLU (or similar) nonlinearities.
=cut

package AI::MXNet::MSRAPrelu;
use Mouse;
extends 'AI::MXNet::Xavier';

has '+rnd_type'    => (default => "gaussian");
has '+factor_type' => (default => "avg");
has 'slope'        => (is => 'ro', isa => 'Num', default => 0.25);

sub BUILD
{
    my $self = shift;
    my $magnitude = 2 / (1 + $self->slope ** 2);
    $self->magnitude($magnitude);
}
1;
