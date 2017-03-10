package AI::MXNet::InitDesc;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::InitDesc

=head1 DESCRIPTION

     Descriptor for initialization pattern.
=cut

=head2 new

    Parameters
    ---------
    name : str
        name of variable
    attrs : hash ref of str to str
        attributes of this variable taken from AI::MXNet::Symbol->attr_dict
=cut
has 'name'   => (is => 'ro', isa => 'Str', required => 1);
has 'attrs'  => (is => 'rw', isa => 'HashRef[Str]', lazy => 1, default => sub { +{} });
use overload '""' => sub { shift->name };
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(name => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

# Base class for Initializers
package AI::MXNet::Initializer;
use Mouse;
use AI::MXNet::Base qw(:DEFAULT pzeros pceil);
use AI::MXNet::NDArray;
use JSON::PP;
use overload "&{}" => sub { my $self = shift; sub { $self->call(@_) } },
             '""'  => sub {
                my $self = shift;
                my ($name) = ref($self) =~ /::(\w+)$/;
                encode_json(
                    [lc $name,
                        $self->kwargs//{ map { $_ => "".$self->$_ } $self->meta->get_attribute_list }
                ]);
             },
             fallback => 1;
has 'kwargs' => (is => 'rw', init_arg => undef, isa => 'HashRef');

=head1 NAME

    AI::MXNet::Initializer

=head1 DESCRIPTION

     Base class for all Initializers
=cut

=head2 register

    Register initializer class to the initializer factory
=cut

my %init_registry;
method get_init_registry()
{
    return \%init_registry;
}

method register()
{
    my ($name) = $self =~ /::(\w+)$/;
    my $orig_name = $name;
    $name         = lc $name;
    if(exists $init_registry{ $name })
    {
        my $existing = $init_registry{ $name };
        warn(
            "WARNING: New initializer $self.$name" 
            ."is overriding existing initializer $existing.$name"
        );
    }
    $init_registry{ $name } = $self;
    {
        no strict 'refs';
        no warnings 'redefine';
        *{"$orig_name"} = sub { shift; $self->new(@_) };
        *InitDesc       = sub { shift; AI::MXNet::InitDesc->new(@_) };
    }
}

=head2 init

        Parameters
        ----------
        desc : AI::MXNet::InitDesc|str
            name of corresponding ndarray
            object that describes the initializer

        arr : NDArray
            ndarray to be Initialized
=cut
use Data::Dumper;
method call(Str|AI::MXNet::InitDesc $desc, AI::MXNet::NDArray $arr)
{
    return $self->_legacy_init($desc, $arr) unless blessed $desc;
    my $init = $desc->attrs->{ __init__ };
    if($init)
    {
      my ($klass, $kwargs) = @{ decode_json($init) };
      $self->get_init_registry->{ lc $klass }->new(%{ $kwargs })->_init_weight("$desc", $arr);
    }
    else
    {
        $desc = "$desc";
        if($desc =~ /(weight|bias|gamma|beta)$/)
        {
            my $method = "_init_$1";
            $self->$method($desc, $arr);
        }
        else
        {
            $self->_init_default($desc, $arr)
        }
    }
}


method _legacy_init(Str $name, AI::MXNet::NDArray $arr)
{
    warnings::warnif(
        'deprecated',
        'Calling initializer with init($str, $NDArray) has been deprecated.'.
        'please use init(mx->init->InitDesc(...), NDArray) instead.'
    );
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
    confess(
        "Unknown initialization pattern for $name. "
        .'Default initialization is now limited to '
        .'"weight", "bias", "gamma" (1.0), and "beta" (0.0).'
        .'Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern'
    );
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

has 'param'        => (is => "rw", isa => 'HashRef[AI::MXNet::NDArray]', required => 1);
has 'default_init' => (is => "rw", isa => "AI::MXNet::Initializer");
has 'verbose'      => (is => "rw", isa => "Int", default => 0);

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

package AI::MXNet::Zero;
use Mouse;
extends 'AI::MXNet::Initializer';
method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    $arr .= 0;
}

__PACKAGE__->register;

package AI::MXNet::One;
use Mouse;
extends 'AI::MXNet::Initializer';
method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    $arr .= 1;
}

__PACKAGE__->register;

package AI::MXNet::Constant;
use Mouse;
extends 'AI::MXNet::Initializer';
has 'value' => (is => 'ro', isa => 'Num', required => 1);
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(value => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    $arr .= $self->value;
}

__PACKAGE__->register;

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
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(scale => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    AI::MXNet::Random->uniform(-$self->scale, $self->scale, { out => $arr });
}

__PACKAGE__->register;
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
around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    return $class->$orig(sigma => $_[0]) if @_ == 1;
    return $class->$orig(@_);
};

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    AI::MXNet::Random->normal(0, $self->sigma, { out => $arr });
}

__PACKAGE__->register;

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
has "rand_type" => (is => "ro", isa => enum([qw/uniform normal/]), default => 'uniform');

method _init_weight(Str $name, AI::MXNet::NDArray $arr)
{
    my @shape = @{ $arr->shape };
    my $nout = $shape[0];
    my $nin = AI::MXNet::NDArray->size([@shape[1..$#shape]]);
    my $tmp = AI::MXNet::NDArray->zeros([$nout, $nin]);
    if($self->rand_type eq 'uniform')
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
__PACKAGE__->register;

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
has "magnitude"   => (is => "rw", isa => "Num", default => 3);
has "rnd_type"    => (is => "ro", isa => enum([qw/uniform gaussian/]), default => 'uniform');
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
__PACKAGE__->register;

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
    $self->kwargs({ slope => $self->slope, factor_type => $self->factor_type });
}
__PACKAGE__->register;

package AI::MXNet::Bilinear;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Initializer';

method _init_weight($name, $arr)
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

__PACKAGE__->register;

package AI::MXNet::FusedRNN;
use Mouse;
use JSON::PP;
extends 'AI::MXNet::Initializer';

=head1 NAME

AI::MXNet::FusedRNN

=head1 DESCRIPTION

    Initialze parameters for fused rnn layer

    Parameters
    ----------
    init : Initializer
        intializer applied to unpacked weights.
    num_hidden : int
        should be the same with arguments passed to FusedRNNCell.
    num_layers : int
        should be the same with arguments passed to FusedRNNCell.
    mode : str
        should be the same with arguments passed to FusedRNNCell.
    bidirectional : bool
        should be the same with arguments passed to FusedRNNCell.
=cut

has 'init'          => (is => 'rw', isa => 'Str|AI::MXNet::Initializer', required => 1);
has [qw/num_hidden
       num_layers/] => (is => 'ro', isa => 'Int', required => 1);
has 'mode'          => (is => 'ro', isa => 'Str', required => 1);
has 'bidirectional' => (is => 'ro', isa => 'Bool', default => 0);

sub BUILD
{
    my $self = shift;
    if(not blessed $self->init)
    {
        my ($klass, $kwargs);
        eval {
            ($klass, $kwargs) = @{ decode_json($self->init) };
        };
        confess("FusedRNN failed to init $@") if $@;
        $self->init($self->get_init_registry->{ lc $klass }->new(%$kwargs));
    }
}

method _init_weight($name, $arr)
{
    my $cell = AI::MXNet::RNN::FusedCell->new(
        num_hidden    => $self->num_hidden,
        num_layers    => $self->num_layers,
        mode          => $self->mode,
        bidirectional => $self->bidirectional,
        prefix        => ''
    );

    my $args = $cell->unpack_weights({ parameters => $arr });
    for my $name (keys %{ $args })
    {
       my $desc = AI::MXNet::InitDesc->new(name => $name);
       &{$self->init}($desc, $args->{name});
    }

    $arr .= $cell->pack_weights($args)->{parameters};
}

1;
