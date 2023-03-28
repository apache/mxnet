# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

package AI::MXNet::Base;
use strict;
use warnings;
use PDL;
use PDL::Types ();
use PDL::CCS::Nd;
use AI::MXNetCAPI 1.32;
use AI::NNVMCAPI 1.3;
use AI::MXNet::Types;
use Time::HiRes;
use Scalar::Util qw(blessed);
use Carp;
use Exporter;
use base qw(Exporter);
use List::Util qw(shuffle);
use Data::Dumper;

@AI::MXNet::Base::EXPORT = qw(product enumerate assert zip check_call build_param_doc
                              pdl cat dog svd bisect_left pdl_shuffle as_array ascsr rand_sparse
                              DTYPE_STR_TO_MX DTYPE_MX_TO_STR DTYPE_MX_TO_PDL
                              DTYPE_PDL_TO_MX DTYPE_MX_TO_PERL GRAD_REQ_MAP
                              STORAGE_TYPE_UNDEFINED STORAGE_TYPE_DEFAULT
                              STORAGE_TYPE_ROW_SPARSE STORAGE_TYPE_CSR
                              STORAGE_TYPE_STR_TO_ID STORAGE_TYPE_ID_TO_STR STORAGE_AUX_TYPES);
@AI::MXNet::Base::EXPORT_OK = qw(pzeros pceil pones digitize hash array_index range);

use constant DTYPE_STR_TO_MX => {
    float32 => 0,
    float64 => 1,
    float16 => 2,
    uint8   => 3,
    int32   => 4,
    int8    => 5,
    int64   => 6
};
use constant DTYPE_MX_TO_STR => {
    0 => 'float32',
    1 => 'float64',
    2 => 'float16',
    3 => 'uint8',
    4 => 'int32',
    5 => 'int8',
    6 => 'int64'
};
use constant DTYPE_MX_TO_PDL => {
    0 => $PDL::Types::PDL_F,
    1 => $PDL::Types::PDL_D,
    2 => $PDL::Types::PDL_F,
    3 => $PDL::Types::PDL_B,
    4 => $PDL::Types::PDL_L,
    5 => $PDL::Types::PDL_SB,
    6 => $PDL::Types::PDL_LL,
    float32 => $PDL::Types::PDL_F,
    float64 => $PDL::Types::PDL_D,
    float16 => $PDL::Types::PDL_F,
    uint8   => $PDL::Types::PDL_B,
    int32   => $PDL::Types::PDL_L,
    int8    => $PDL::Types::PDL_SB,
    int64   => $PDL::Types::PDL_LL,
};
use constant DTYPE_PDL_TO_MX => {
    $PDL::Types::PDL_F  => 0,
    $PDL::Types::PDL_D  => 1,
    $PDL::Types::PDL_B  => 3,
    $PDL::Types::PDL_L  => 4,
    $PDL::Types::PDL_LL => 6,
};
use constant DTYPE_MX_TO_PERL => {
    0 => 'f',
    1 => 'd',
    2 => 'S',
    3 => 'C',
    4 => 'l',
    5 => 'c',
    6 => 'q',
    float32 => 'f',
    float64 => 'd',
    float16 => 'S',
    uint8   => 'C',
    int32   => 'l',
    int8    => 'c',
    int64   => 'q'
};
use constant GRAD_REQ_MAP => {
    null  => 0,
    write => 1,
    add   => 3
};
use constant {
    STORAGE_TYPE_UNDEFINED  => -1,
    STORAGE_TYPE_DEFAULT    =>  0,
    STORAGE_TYPE_ROW_SPARSE =>  1,
    STORAGE_TYPE_CSR        =>  2
};
use constant STORAGE_TYPE_STR_TO_ID => {
    undefined  => STORAGE_TYPE_UNDEFINED,
    default    => STORAGE_TYPE_DEFAULT,
    row_sparse => STORAGE_TYPE_ROW_SPARSE,
    csr        => STORAGE_TYPE_CSR
};
use constant STORAGE_TYPE_ID_TO_STR => {
    STORAGE_TYPE_UNDEFINED()  => 'undefined',
    STORAGE_TYPE_DEFAULT()    => 'default',
    STORAGE_TYPE_ROW_SPARSE() => 'row_sparse',
    STORAGE_TYPE_CSR()        => 'csr'
};
use constant STORAGE_AUX_TYPES => {
    row_sparse => ['int64'],
    csr => ['int64', 'int64']
};


=head1 NAME

    AI::MXNet::Base - Helper functions

=head1 DEFINITION

    Helper functions

=head2 zip

    Perl version of for x,y,z in zip (arr_x, arr_y, arr_z)

    Parameters
    ----------
    $sub_ref, called with @_ filled with $arr_x->[$i], $arr_y->[$i], $arr_z->[$i]
    for each loop iteration.

    @array_refs
=cut

sub zip
{
    if('CODE' eq ref $_[0])
    {
        # continue supporting the callback style
        my $code = shift;
        $code->(@$_) for AI::MXNetCAPI::py_zip(map { \@$_ } @_);
        return;
    }
    # the map() here may seem like a no-op, but triggers overloading or
    # whatever else is needed to make array-ish things actually arrays
    # before entering the low level list builder.
    return AI::MXNetCAPI::py_zip(map { \@$_ } @_);
}

=head2 enumerate

    Same as zip, but the argument list in the anonymous sub is prepended
    by the iteration count.
=cut

sub enumerate
{
    if('CODE' eq ref $_[0])
    {
        # continue supporting the callback style
        my $code = shift;
        my $len = @{ $_[0] };
        $code->(@$_) for AI::MXNetCAPI::py_zip([0..$len-1], map { \@$_ } @_);
        return;
    }
    my $len = @{ $_[0] };
    return AI::MXNetCAPI::py_zip([0..$len-1], map { \@$_ } @_);
}

=head2 product

    Calculates the product of the input agruments.
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

=head2 pdl_shuffle

    Shuffle the pdl by the last dimension

    Parameters
    -----------
    PDL $pdl
    $preshuffle Maybe[ArrayRef[Index]], if defined the array elements are used
    as shuffled last dimension's indexes
=cut

sub pdl_shuffle
{
    my ($pdl, $preshuffle) = @_;
    my @shuffle = $preshuffle ? @{ $preshuffle } : shuffle(0..$pdl->dim(-1)-1);
    return $pdl->dice_axis(-1, pdl(\@shuffle));
}

=head2 assert

    Parameters
    -----------
    Bool $input
    Str  $error_str
    Calls Carp::confess with $error_str//"AssertionError" if the $input is false
=cut

sub assert
{
    my ($input, $error_str) = @_;
    local($Carp::CarpLevel) = 1;
    Carp::confess($error_str//'AssertionError')
        unless $input;
}

=head2 check_call

    Checks the return value of C API call

    This function will raise an exception when error occurs.
    Every API call is wrapped with this function.

    Returns the C API call return values stripped of first return value,
    checks for return context and returns first element in
    the values list when called in scalar context.
=cut

sub check_call
{
    Carp::confess(AI::MXNetCAPI::GetLastError()) if shift;
    return wantarray ? @_ : $_[0];
}

=head2 build_param_doc

    Builds argument docs in python style.

    arg_names : array ref of str
        Argument names.

    arg_types : array ref of str
        Argument type information.

    arg_descs : array ref of str
        Argument description information.

    remove_dup : boolean, optional
        Whether to remove duplication or not.

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
    for(zip($arg_names, $arg_types, $arg_descs)) {
            my ($key, $type_info, $desc) = @$_;
            next if exists $param_keys{$key} and $remove_dup;
            $param_keys{$key} = 1;
            my $ret = sprintf("%s : %s", $key, $type_info);
            $ret .= "\n    ".$desc if length($desc);
            push @param_str,  $ret;
    }
    return sprintf("Parameters\n----------\n%s\n", join("\n", @param_str));
}

=head2 _notify_shutdown

    Notify MXNet about shutdown.
=cut

sub _notify_shutdown
{
    check_call(AI::MXNetCAPI::NotifyShutdown());
}

sub _indent
{
    my ($s_, $numSpaces) = @_;
    my @s = split(/\n/, $s_);
    if (@s == 1)
    {
        return $s_;
    }
    my $first = shift(@s);
    @s = ($first, map { (' 'x$numSpaces) . $_ } @s);
    return join("\n", @s);
}

sub as_array
{
    return ref $_[0] eq 'ARRAY' ? $_[0] : [$_[0]];
}

my %internal_arguments = (prefix => 1, params => 1, shared => 1);
my %attributes_per_class;
sub process_arguments
{
    my $orig  = shift;
    my $class = shift;
    if($class->can('python_constructor_arguments'))
    {
        if(not exists $attributes_per_class{$class})
        {
            %{ $attributes_per_class{$class} } = map { $_->name => 1 } $class->meta->get_all_attributes;
        }
        my %kwargs;
        while(@_ >= 2 and defined $_[-2] and not ref $_[-2] and (exists $attributes_per_class{$class}{ $_[-2] } or exists $internal_arguments{ $_[-2] }))
        {
            my $v = pop(@_);
            my $k = pop(@_);
            $kwargs{ $k } = $v;
        }
        if(@_)
        {
            my @named_params = @{ $class->python_constructor_arguments };
            Carp::confess("Paramers mismatch expected ".Dumper(\@named_params).", but got ".Dumper(\@_))
                if @_ > @named_params;
            @kwargs{ @named_params[0..@_-1] } = @_;
        }
        return $class->$orig(%kwargs);
    }
    return $class->$orig(@_);
}

END {
    _notify_shutdown();
    Time::HiRes::sleep(0.01);
}

*pzeros = \&zeros;
*pones = \&ones;
*pceil  = \&ceil;
## making sure that we can stringify arbitrarily large piddles
$PDL::toolongtoprint = 1000_000_000;
## convenience subs

sub ascsr
{
    my ($data, $indptr, $indices, $shape) = @_;
    my @which;
    my $i = 0;
    my $j = 0;
    while($i < $indices->nelem)
    {
        for($i = $indptr->at($j); $i < $indptr->at($j+1); $i++)
        {
            push @which, [$j, $indices->at($i)];
        }
        $j++;
    }
    return PDL::CCS::Nd->newFromWhich(
            pdl(\@which), $data, pdims => blessed $shape ? $shape : pdl($shape)
    )->xchg(0, 1);
}

package AI::MXNet::COO::Nd;
use Mouse;
has ['data', 'row', 'col'] => (is => 'rw');
no Mouse;

package AI::MXNet::Base;

sub tocoo
{
    my $csr = shift;
    return AI::MXNet::COO::Nd->new(
        data => $csr->data,
        row  => $csr->_whichND->slice(0)->flat,
        col  => $csr->_whichND->slice(1)->flat
    );
}

sub rand_sparse
{
    my ($num_rows, $num_cols, $density, $dtype, $format) = @_;
    $dtype  //= 'float32';
    $format //= 'csr';
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    my $dense = random($pdl_type, $num_cols, $num_rows);
    my $missing = 0;
    $dense->where(random($num_cols, $num_rows)<=1-$density) .= $missing;
    if($format eq 'csr')
    {
        return $dense->tocsr;
    }
    return $dense;
}

{
    no warnings 'once';
    *PDL::CCS::Nd::data    = sub { shift->_nzvals };
    *PDL::CCS::Nd::indptr  = sub { my $self = shift; ($self->hasptr ? $self->getptr : $self->ptr)[0] };
    *PDL::CCS::Nd::indices = sub { shift->_whichND->slice(1)->flat };
    *PDL::CCS::Nd::tocoo   = sub { tocoo(shift) };
    *PDL::CCS::Nd::shape   = sub { shift->pdims };
    *PDL::CCS::Nd::dtype   = sub { DTYPE_MX_TO_STR->{ DTYPE_PDL_TO_MX->{ shift->type->numval } } };
    *PDL::tocsr            = sub { shift->xchg(0, 1)->toccs->xchg(0, 1) };
    *PDL::rand_sparse      = sub { shift; rand_sparse(@_) };
}

{
    my $orig_at = PDL->can('at');
    no warnings 'redefine';
    *PDL::at = sub {
        my ($self, @args) = @_;
        return $orig_at->($self, @args) if @args != 1;
        return $orig_at->($self, @args) if $self->ndims == 1;
        return $self->slice(('X')x($self->ndims-1), $args[0])->squeeze;
    };
    *PDL::len    = sub { shift->dim(-1) };
    *PDL::dtype  = sub { DTYPE_MX_TO_STR->{ DTYPE_PDL_TO_MX->{ shift->type->numval } } };
}

sub digitize
{
    my ($d, $bins) = @_;
    for(my $i = 0; $i < @$bins; $i++)
    {
        return $i if $d < $bins->[$i];
    }
    return scalar(@$bins);
}

use B;
sub hash { hex(B::hash(shift)) }
use List::Util ();
sub array_index { my ($s, $array) = @_; return List::Util::first { $array->[$_] eq $s } 0..@$array-1 }
sub range { my ($begin, $end, $step) = @_; $step //= 1; grep { not (($_-$begin) % $step) } $begin..$end-1 }

1;
