package AI::MXNet::Base;
use strict;
use warnings;
use PDL;
use PDL::Types qw();
use AI::MXNetCAPI 0.02;
use AI::NNVMCAPI 0.02;
use AI::MXNet::Types;
use Time::HiRes;
use Carp;
use Exporter;
use base qw(Exporter);
@AI::MXNet::Base::EXPORT = qw(product enumerate assert zip check_call build_param_doc 
                              pdl cat dog svd bisect_left
                              DTYPE_STR_TO_MX DTYPE_MX_TO_STR DTYPE_MX_TO_PDL
                              DTYPE_PDL_TO_MX DTYPE_MX_TO_PERL);
@AI::MXNet::Base::EXPORT_OK = qw(pzeros pceil);
use constant DTYPE_STR_TO_MX => {
    float32 => 0,
    float64 => 1,
    float16 => 2,
    uint8   => 3,
    int32   => 4
};
use constant DTYPE_MX_TO_STR => {
    0 => 'float32',
    1 => 'float64',
    2 => 'float16',
    3 => 'uint8',
    4 => 'int32'
};
use constant DTYPE_MX_TO_PDL => {
    0 => 6,
    1 => 7,
    2 => 6,
    3 => 0,
    4 => 3,
    float32 => 6,
    float64 => 7,
    float16 => 6,
    uint8   => 0,
    int32   => 3
};
use constant DTYPE_PDL_TO_MX => {
    6 => 0,
    7 => 1,
    0 => 3,
    3 => 4,
};
use constant DTYPE_MX_TO_PERL => {
    0 => 'f',
    1 => 'd',
    2 => 'S',
    3 => 'C',
    4 => 'l',
    float32 => 'f',
    float64 => 'd',
    float16 => 'S',
    uint8   => 'C',
    int32   => 'l'
};


=head1 Definition

Helper functions

=head2 zip

    Perl version of for x,y,z in zip (arr_x, arr_y, arr_z)

    Parameters
    ----------
    $sub_ref, called with @_ filled with 
    $arr_x->[$i], $arr_y->[$i], $arr_z->[$i]
    for each loop iteration.

    @array_refs
=cut

sub zip
{
    my ($sub, @arrays) = @_;
    my $len = @{ $arrays[0] };
    for (my $i = 0; $i < $len; $i++)
    {
        $sub->(map { $_->[$i] } @arrays);
    }
}

=head2 enumerate

    Same as zip, but argument list in anonymous sub is prepended
    by iteration count
=cut

sub enumerate
{
    my ($sub, @arrays) = @_;
    my $len = @{ $arrays[0] };
    zip($sub, [0..$len-1], @arrays);
}

=head2 product

    Calculates product of the input agruments
=cut

sub product
{
    my $p = 1;
    map { $p = $p * $_ } @_;
    return $p;
}

=head2 bisect_left

    https://hg.python.org/cpython/file/2.7/Lib/bisect.py

=cut

sub bisect_left
{
    my ($a, $x, $lo, $hi) = @_;
    $lo //= 0;
    $hi //= @{ $a };
    if($lo < 0)
    {
        Carp::confess('lo must be non-negative');
    }
    while($lo < $hi)
    {
        my $mid = int(($lo+$hi)/2);
        if($a->[$mid] < $x)
        {
            $lo = $mid+1;
        }
        else
        {
            $hi = $mid;
        }
    }
    return $lo;
}


=head2 assert

    Parameters
    -----------
    Bool $input
    Str  $error_str
    Calls Carp::confess with $error_str//"AssertionError" if $input is false
=cut

sub assert
{
    my ($input, $error_str) = @_;
    local($Carp::CarpLevel) = 1;
    Carp::confess($error_str//'AssertionError')
        unless $input;
}

=head2 check_call

    Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    returns C API call return values stripped of first return value,
    checks for return context and returns first element in
    the values list when called in scalar context. 
=cut

sub check_call
{
    Carp::confess(AI::MXNetCAPI::GetLastError()) if shift;
    return wantarray ? @_ : $_[0];
}

=head2 build_param_doc

    Build argument docs in python style.

    arg_names : list of str
        Argument names.

    arg_types : list of str
        Argument type information.

    arg_descs : list of str
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
=cut

sub build_param_doc
{
    my ($arg_names, $arg_types, $arg_descs, $remove_dup) = @_;
    $remove_dup //= 1;
    my %param_keys;
    my @param_str;
    zip(sub { 
            my ($key, $type_info, $desc) = @_;
            return if exists $param_keys{$key} and $remove_dup;
            $param_keys{$key} = 1;
            my $ret = sprintf("%s : %s", $key, $type_info);
            $ret .= "\n    ".$desc if length($desc); 
            push @param_str,  $ret;
        },
        $arg_names, $arg_types, $arg_descs
    );
    return sprintf("Parameters\n----------\n%s\n", join("\n", @param_str));
}

=head2 _notify_shutdown

    Notify MXNet about a shutdown.
=cut

sub _notify_shutdown
{
    check_call(AI::MXNetCAPI::NotifyShutdown());
}

END {
    _notify_shutdown();
    Time::HiRes::sleep(0.01);
}

*pzeros = \&zeros;
*pceil  = \&ceil;
## making sure that we can stringify arbitrarily large piddles
$PDL::toolongtoprint = 1000_000_000;

1;
