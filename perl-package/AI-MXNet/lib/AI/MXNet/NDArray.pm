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

package AI::MXNet::NDArray;

=head1 NAME

    AI::MXNet::NDArray - Multidimensional tensor object of MXNet.
=cut

use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray::Slice;
use AI::MXNet::Context;
use Mouse;
use AI::MXNet::Function::Parameters;
use overload
    '""' => \&stringify,
    '+'  => \&add,
    '+=' => \&iadd,
    '-'  => \&subtract,
    '-=' => \&isubtract,
    '*'  => \&multiply,
    '*=' => \&imultiply,
    '/'  => \&divide,
    '/=' => \&idivide,
    '%'  => \&modulo,
    '%=' => \&imodulo,
    '**' => \&power,
    '==' => \&equal,
    '!=' => \&not_equal,
    '>'  => \&greater,
    '>=' => \&greater_equal,
    '<'  => \&lesser,
    '<=' => \&lesser_equal,
    '.=' => \&set,
    '@{}'=> \&split_array,
    '='  => sub { $_[0] };

extends 'AI::MXNet::NDArray::Base';
has 'writable' => (is => 'rw', isa => 'Int', default => 1, lazy => 1);
has 'handle'   => (is => 'rw', isa => 'NDArrayHandle', required => 1);

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::NDArrayFree(shift->handle));
}

method STORABLE_freeze($cloning)
{
    my $buf = check_call(AI::MXNetCAPI::NDArraySaveRawBytes($self->handle));
    return ($buf,\ $self->writable);
}

method STORABLE_thaw($cloning, $buf, $writable)
{
    my $handle = check_call(
                    AI::MXNetCAPI::NDArrayLoadFromRawBytes(
                        $buf, length($buf)
                    )
    );
    $self->handle($handle);
    $self->writable($$writable);
}

method split_array(@args)
{
     $self->shape->[0] > 1 ? $self->split(num_outputs => $self->shape->[0], squeeze_axis => 1, axis => 0) : [$self];
}

method at(Index @indices)
{
    confess("No idxs supplied") unless @indices;
    my $shape = $self->shape;
    my $dsize = @$shape;
    my $isize = @indices;
    confess("Dimensions size $dsize < indexes size $isize")
        if $dsize < $isize;
    confess("Dimensions size $dsize = indexes size $isize,
                   ndarray only supports either ->at on dimension 0
                   or full crop")
        if $isize > 1 and $dsize != $isize;
    my $i = 0;
    for(zip(\@indices, $shape)) {
        my ($idx, $dim_size) = @$_;
        confess("Dimension $i mismatch Idx: $idx >= Dim Size: $dim_size")
            if $idx >= $dim_size or ($idx + $dim_size) < 0;
        ++$i;
    }
    $i = 0;
    for my $v (@indices)
    {
        $v += $shape->[$i] if $v < 0;
        ++$i;
    }
    return $self->_at($indices[0]) if @indices == 1;
    return $self->slice(@indices);
}

method len() { $self->shape->[0] }

method slice(Slice|AdvancedSlice @slices)
{
    confess("No slices supplied") unless @slices;
    if(ref $slices[0] eq 'ARRAY' and ref $slices[0]->[0])
    {
        my @indices;
        my $key = $slices[0];
        my $dtype = 'int32';
        for my $idx_i (@{ $key })
        {
            if(not (blessed $idx_i and $idx_i->isa(__PACKAGE__)))
            {
                $idx_i = __PACKAGE__->array($idx_i, ctx=>$self->context, dtype=>$dtype);
            }
            else
            {
                $dtype = $idx_i->dtype;
            }
            push @indices, $idx_i;
        }
        my $indices = __PACKAGE__->stack(@indices);
        return __PACKAGE__->gather_nd($self, $indices);
    }
    my $shape = $self->shape;
    my $dsize = @$shape;
    my $isize = @slices;
    confess("Dimensions size $dsize < slices size $isize")
        if $dsize < $isize;
    confess("Dimensions size $dsize != slices size $isize,
                   ndarray only supports either ->slice on dimension 0
                   or full crop")
        if $isize > 1 and $dsize != $isize;
    my $i = -1;
    @slices = map {
        ++$i;
        ref $_ ? (@$_ == 1 ? [$_->[0], $_->[0]] : $_) : ($_ eq 'X' ? [0, $shape->[$i] - 1] : [$_, $_]);
    } @slices;
    for(zip(\@slices, $shape)) {
        my ($slice, $dim_size) = @$_;
        my ($begin, $end, $stride) = @$slice;
        confess("NDArray does not support slice strides != 1")
            if ($stride//0) > 1;
        confess("Dimension $i mismatch slice begin : $begin >= Dim Size: $dim_size")
            if $begin >= $dim_size or ($begin + $dim_size) < 0;
        confess("Dimension $i mismatch slice end : $end >= Dim Size: $dim_size")
            if $end >= $dim_size or ($end + $dim_size) < 0;
    }
    $i = 0;
    my ($begin, $end) = ([], []);
    for my $s (@slices)
    {
        $s->[0] += $shape->[$i] if $s->[0] < 0;
        $s->[1] += $shape->[$i] if $s->[1] < 0;
        confess("Dimension $i slice mismatch (begin $s->[0] > end $s->[1])")
            if($s->[0] > $s->[1]);
        push @$begin, $s->[0];
        push @$end, $s->[1] + 1;
        $i++;
    }
    return $self->_slice($begin->[0], $end->[0]) if @slices == 1;
    return AI::MXNet::NDArray::Slice->new(parent => $self, begin => $begin, end => $end);
}

method set(AcceptableInput $value, $reverse=)
{
    confess("set value must be defined") unless defined $value;
    confess("Array is not writable") if not $self->writable;
    ## plain number
    if(not ref $value)
    {
        $self->_set_value($value, out => $self);
    }
    # ndarray
    elsif(blessed($value) and $value->isa(__PACKAGE__))
    {
        $value->copyto($self);
    }
    # slice of another ndarray
    elsif(blessed($value) and $value->isa('AI::MXNet::NDArray::Slice'))
    {
        $value->sever->copyto($self);
    }
    # perl array, PDL, PDL::Matrix
    else
    {
        $self->_sync_copyfrom($value);
    }
    return $self;
}

method asscalar()
{
    confess("ndarray size must be 1") unless $self->size == 1;
    return $self->aspdl->at(0);
    ## code below works happily on CPU/segfaults on GPU
    #$self->wait_to_read;
    #my $perl_pack_type = DTYPE_MX_TO_PERL->{$self->dtype};
    #my $length = {qw/f 4 d 8 S 2 C 1 l 4/}->{$perl_pack_type};
    #return
    #(map {
    #        $perl_pack_type eq 'S' ? AI::MXNetCAPI::_half_to_float($_) : $_
    #     } unpack("$perl_pack_type", check_call(AI::MXNetCAPI::NDArrayGetData($self->handle, $length)))
    #)[0];
}

method _sync_copyfrom(ArrayRef|PDL|PDL::Matrix $source_array)
{
    my $dtype = $self->dtype;
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    if(not blessed($source_array))
    {
        $source_array = eval {
            pdl($pdl_type, $source_array);
        };
        confess($@) if $@;
    }
    if($pdl_type->numval != $source_array->type->numval)
    {
        my $convert_func = $pdl_type->convertfunc;
        $source_array = $source_array->$convert_func;
    }
    $source_array = pdl($pdl_type, [@{ $source_array->unpdl } ? $source_array->unpdl->[0] : 0 ])
        unless @{ $source_array->shape->unpdl };
    my $pdl_shape = $source_array->shape->unpdl;
    my $pdl_shape_str = join(',', ref($source_array) eq 'PDL' ? reverse @{ $pdl_shape } : @{ $pdl_shape });
    my $ndary_shape_str = join(',', @{ $self->shape });
    if($pdl_shape_str ne $ndary_shape_str)
    {
        confess("Shape inconsistant: expected $ndary_shape_str vs got $pdl_shape_str")
    }
    my $perl_pack_type = DTYPE_MX_TO_PERL->{$dtype};
    my $ptr = $source_array->get_dataref;
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $ptr = \( pack("S*", map { AI::MXNetCAPI::_float_to_half($_) } unpack ("f*", $$ptr)) );
    }
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromCPU($self->handle, $$ptr, $self->size));
    return $self;
}

=head2 aspdl

    Returns a copied PDL array of current array.

    Returns
    -------
    array : PDL
        A copy of the array content.
=cut

method aspdl()
{
    my $dtype = $self->dtype;
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    my $pdl = PDL->new_from_specification($pdl_type, reverse @{ $self->shape });
    my $perl_pack_type = DTYPE_MX_TO_PERL->{$dtype};
    my $ptr = $pdl->get_dataref;
    check_call(AI::MXNetCAPI::NDArraySyncCopyToCPU($self->handle, $$ptr, $self->size));
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $$ptr = pack("f*", map { AI::MXNetCAPI::_half_to_float($_) } unpack("S*", $$ptr));
    }
    $pdl->upd_data;
    return $pdl;
}


=head2 asmpdl

    Returns copied PDL::Matrix objectt of current array.

    Requires caller to "use PDL::Matrix" in user space.

    Returns
    -------
    array : PDL::Matrix
        A copy of array content.
=cut

method asmpdl()
{
    my $dtype = $self->dtype;
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    my $pdl = PDL::Matrix->new_from_specification($pdl_type, @{ $self->shape });
    my $perl_pack_type = DTYPE_MX_TO_PERL->{$dtype};
    my $ptr = $pdl->get_dataref;
    check_call(AI::MXNetCAPI::NDArraySyncCopyToCPU($self->handle, $$ptr, $self->size));
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $$ptr = pack("f*", map { AI::MXNetCAPI::_half_to_float($_) } unpack("S*", $$ptr));
    }
    $pdl->upd_data;
    return $pdl;
}


=head2 _slice

    Returns sliced NDArray that shares memory with the current one.

    Parameters
    ----------
    start : int
        Starting index of slice.
    stop : int
        Finishing index of slice.
=cut

method _slice (
    Index $start,
    Index $stop
)
{
    confess("start $start > stop $stop") if $start > $stop;
    my $handle = check_call(
        AI::MXNetCAPI::NDArraySlice(
            $self->handle,
            $start,
            $stop
        )
    );
    return __PACKAGE__->new(handle => $handle, writable => $self->writable);
}

=head2  _at

    Returns a sub NDArray that shares memory with current one.

    Parameters
    ----------
    idx : int
        index of the sub array.
=cut


method _at(Index $idx)
{
    my $handle = check_call(
                AI::MXNetCAPI::NDArrayAt(
                    $self->handle, $idx >=0 ? $idx : $self->shape->[0] + $idx
                )
    );
    return __PACKAGE__->new(handle => $handle, writable => $self->writable);
}

=head2 reshape

    Returns a reshaped NDArray that shares the memory with current one.
    One shape dimension can be -1. In this case, the value is inferred
    from the length of the array and remaining dimensions.

    Parameters
    ----------
    new_shape : Shape
        new shape of NDArray
=cut

method reshape(ArrayRef[Int] $new_shape)
{
    my $i = -1;
    my @inferred = map { $i++; $_ == -1 ? ($i) : () } @$new_shape;
    assert((@inferred <= 1), 'Only one dimension can be inferred.');
    $i = -1;
    my @keep = map { $i++; $_ == 0 ? ($i) : () } @$new_shape;
    my $shape = $self->shape;
    if(@keep)
    {
        @{$new_shape}[@keep] = @{$shape}[@keep];
    }
    if(@inferred)
    {
        $new_shape->[$inferred[0]] = product(@{ $shape })/product(map { abs($_) } @{ $new_shape });
    }
    my $handle = check_call(
                    AI::MXNetCAPI::NDArrayReshape(
                        $self->handle,
                        scalar(@$new_shape),
                        $new_shape
                    )
    );
    return __PACKAGE__->new(handle => $handle, writable => $self->writable);
}

=head2 ndim

    Returns the number of dimensions of this array.
=cut

method ndim()
{
    scalar(@{ $self->shape });
}

=head2 moveaxis

    Moves the 'source' axis into the 'destination' position
    while leaving the other axes in their original order

    Parameters
    ----------
    source : int
        Original position of the axes to move.
    destination : int
        Destination position for each of the original axes.

    Returns
    -------
    result :NDArray
    Array with moved axes.

    Examples
    --------
    > $X = mx->nd->array([[1, 2, 3],
                          [4, 5, 6]]);
    > print Dumper($X->moveaxis(0, 1)->shape)
    > [3, 2]
=cut

method moveaxis(Int $source, Int $dest)
{
    my @axes = 0..$self->ndim-1;
    $source += @axes if $source < 0;
    $dest += @axes if $dest < 0;
    assert($source < @axes);
    assert($dest < @axes);
    my ($to_move) = splice(@axes, $source, 1);
    splice(@axes, $dest, 0, $to_move);
    return __PACKAGE__->transpose($self, \@axes);
}

=head2 broadcast_to

    Broadcasting the current NDArray into the given shape.

    Parameters
    ---------
    Shape $shape : the shape to broadcast
=cut

method broadcast_to(Shape $shape)
{
    my $cur_shape = $self->shape;
    my $err_str = "operands could not be broadcast together with remapped shapes"
                  ."[original->remapped]: [@$cur_shape] and requested shape [@$shape]";
    if(@$shape < @$cur_shape)
    {
        confess($err_str);
    }
    @$cur_shape = ((1)x(@$shape - @$cur_shape), @$cur_shape);
    my $cur_shape_arr = pdl($cur_shape);
    my $broadcasting_axes = ($cur_shape_arr != pdl($shape))->which->unpdl;
    if (grep { $cur_shape->[$_] != 1 } @$broadcasting_axes)
    {
        confess($err_str);
    }
    if(join(',',@$cur_shape) ne join(',',@{ $self->shape }))
    {
        return __PACKAGE__->SUPER::broadcast_to($self->reshape($cur_shape),{ shape => $shape });
    }
    else
    {
        return __PACKAGE__->SUPER::broadcast_to($self, { shape => $shape });
    }
}

=head2 wait_to_read

    Block until all pending write operations on the NDArray are finished.

    This function will return when all the pending writes to the current
    NDArray are finished. There can be pending reads going on when the
    function returns.
=cut

method wait_to_read()
{
    check_call(AI::MXNetCAPI::NDArrayWaitToRead($self->handle));
}

=head2 shape

    Get the shape of current NDArray.

    Returns
    -------
    an array ref representing the shape of current ndarray
=cut

method shape()
{
    return scalar(check_call(AI::MXNetCAPI::NDArrayGetShape($self->handle)));
}

=head2 size

    Number of elements in the array.
=cut

method size(Shape|Undef $shape=)
{
    my $size = 1;
    map { $size *= $_ } @{ $shape//$self->shape };
    return $size;
}


=head2 context

    The context of the NDArray.

    Returns
    -------
    $context : AI::MXNet::Context
=cut

method context()
{
    my ($dev_type_id, $dev_id) = check_call(
        AI::MXNetCAPI::NDArrayGetContext($self->handle)
    );
    return AI::MXNet::Context->new(
        device_type => AI::MXNet::Context::devtype2str->{ $dev_type_id },
        device_id => $dev_id
    );
}

=head2 dtype

    The data type of current NDArray.

    Returns
    -------
    a data type string ('float32', 'float64', 'float16', 'uint8', 'int32')
    representing the data type of the ndarray.
    'float32' is the default dtype for the ndarray class.
=cut

method dtype()
{
    my $dtype = check_call(
        AI::MXNetCAPI::NDArrayGetDType(
            $self->handle
        )
    );
    return DTYPE_MX_TO_STR->{ $dtype };
}

=head2 copyto

    Copy the content of current array to another entity.

    When another entity is the NDArray, the content is copied over.
    When another entity is AI::MXNet::Context, a new NDArray in the context
    will be created.

    Parameters
    ----------
    other : NDArray or Context
        Target NDArray or context we want to copy data to.

    Returns
    -------
    dst : NDArray
=cut

method copyto(AI::MXNet::Context|AI::MXNet::NDArray $other)
{
    if(blessed($other) and $other->isa('AI::MXNet::Context'))
    {
        my $hret = __PACKAGE__->empty(
            $self->shape,
            ctx => $other,
            dtype => $self->dtype
        );
        return __PACKAGE__->_copyto($self, { out => $hret });
    }
    else
    {
        if ($other->handle eq $self->handle)
        {
            Carp::cluck('copy an array to itself, is it intended?');
        }
        return __PACKAGE__->_copyto($self, { out => $other });
    }
}

=head2 copy

    Makes a copy of the current ndarray in the same context

    Returns
    ------
    $copy : NDArray
=cut

method copy()
{
    return $self->copyto($self->context);
}

## alias for PDL::NiceSlice
*sever = \&copy;

=head2 T

    Get transpose of the NDArray.
    Works only on 2-D matrices.
=cut

method T()
{
    if (@{$self->shape} > 2)
    {
        confess('Only 2D matrix is allowed to be transposed');
    }
    return __PACKAGE__->transpose($self);
}

=head2 astype

    Returns copied ndarray of current array with the specified type.

    Parameters
    ----------
    $dtype : Dtype

    Returns
    -------
    $array : ndarray
        A copy of the array content.
=cut

method astype(Dtype $dtype)
{
    my $res = __PACKAGE__->empty($self->shape, ctx => $self->context, dtype => $dtype);
    $self->copyto($res);
    return $res;
}

=head2 as_in_context

    Returns an NDArray in the target context.
    If the array is already in that context, self is returned. Otherwise, a copy is
    made.

    Parameters
    ----------
    context : AI::MXNet::Context
        The target context we want the return value to live in.

    Returns
    -------
        A copy or self as an NDArray in the target context.
=cut

method as_in_context(AI::MXNet::Context $context)
{
    return $self if $self->context == $context;
    return $self->copyto($context);
}

=head2 onehot_encode

    One hot encoding indices into matrix out.

    Parameters
    ----------
    indices: NDArray
        An NDArray containing indices of the categorical features.

    out: NDArray
        The result of the encoding.

    Returns
    -------
        $out: NDArray
=cut

method onehot_encode(AI::MXNet::NDArray $indices, AI::MXNet::NDArray $out)
{
    return __PACKAGE__->_onehot_encode($indices, $out, { out => $out });
}

=head2 _ufunc_helper(lhs, rhs, fn_array, lfn_scalar, rfn_scalar):

    Helper function for element-wise operation
    The function will perform numpy-like broadcasting if needed and call different functions

    Parameters
    ----------
    lhs : NDArray or numeric value
        left hand side operand

    rhs : NDArray or numeric value
        right hand side operand

    fn_array : function
        function to be called if both lhs and rhs are of NDArray type

    lfn_scalar : function
        function to be called if lhs is NDArray while rhs is numeric value

    rfn_scalar : function
        function to be called if lhs is numeric value while rhs is NDArray;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    -------
    out: NDArray
        result array
=cut

sub  _ufunc_helper
{
    my ($lhs, $rhs, $fn_array, $lfn_scalar, $rfn_scalar, $reverse) = @_;
    ($rhs, $lhs) = ($lhs, $rhs) if $reverse and $rfn_scalar;
    if(not ref $lhs)
    {
        if(not $rfn_scalar)
        {
            return __PACKAGE__->can($lfn_scalar)->(__PACKAGE__, $rhs, $lhs);
        }
        else
        {
            return __PACKAGE__->can($rfn_scalar)->(__PACKAGE__, $rhs, $lhs);
        }
    }
    elsif(not ref $rhs)
    {
        return __PACKAGE__->can($lfn_scalar)->(__PACKAGE__, $lhs, $rhs);
    }
    else
    {
        return __PACKAGE__->can($fn_array)->(__PACKAGE__, $lhs, $rhs);
    }
}

method stringify($other=, $reverse=)
{
    sprintf("<%s %s @%s>", ref($self), join('x', @{ $self->shape }), $self->context);
}

method iadd(AI::MXNet::NDArray|Num $other, $reverse=)
{
    confess('trying to add to a readonly NDArray') unless $self->writable;
    return ref $other
        ? __PACKAGE__->broadcast_add($self, $other, { out => $self })
        : __PACKAGE__->_plus_scalar($self, $other, { out => $self })
}

method add(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_add _plus_scalar/
    );
}


method subtract(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_sub _minus_scalar _rminus_scalar/,
        $reverse
    );
}

method isubtract(AI::MXNet::NDArray|Num $other, $reverse=)
{
    confess('trying to add to a readonly NDArray') unless $self->writable;
    return ref $other
        ? __PACKAGE__->broadcast_sub($self, $other, { out => $self })
        : __PACKAGE__->_minus_scalar($self, $other, { out => $self })
}

method multiply(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_mul _mul_scalar/
    );
}

method imultiply(AI::MXNet::NDArray|Num $other, $reverse=)
{
    confess('trying to add to a readonly NDArray') unless $self->writable;
    return ref $other
        ? __PACKAGE__->broadcast_mul($self, $other, { out => $self })
        : __PACKAGE__->_mul_scalar($self, $other, { out => $self })
}

method divide(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_div _div_scalar _rdiv_scalar/,
        $reverse
    );
}

method idivide(AI::MXNet::NDArray|Num $other, $reverse=)
{
    confess('trying to add to a readonly NDArray') unless $self->writable;
    return ref $other
        ? __PACKAGE__->broadcast_div($self, $other, { out => $self })
        : __PACKAGE__->_div_scalar($self, $other, { out => $self })
}

method power(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_power _power_scalar _rpower_scalar/,
        $reverse
    );
}

method maximum(AI::MXNet::NDArray|Num $other)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_maximum _maximum_scalar/
    );
}

method minimum(AI::MXNet::NDArray|Num $other)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_minimum _minimum_scalar/
    );
}

method equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_equal _equal_scalar/
    );
}

method not_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_not_equal _not_equal_scalar/
    );
}

method greater(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_greater _greater_scalar _lesser_scalar/,
        $reverse
    );
}

method greater_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_greater_equal _greater_equal_scalar _lesser_equal_scalar/,
        $reverse
    );
}

method lesser(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_lesser _lesser_scalar _greater_scalar/,
        $reverse
    );
}

method lesser_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_lesser_equal _lesser_equal_scalar _greater_equal_scalar/,
        $reverse
    );
}

method true_divide(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return $self->divide($other, $reverse);
}

method modulo(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_mod _mod_scalar _rmod_scalar/,
        $reverse
    );
}

method imodulo(AI::MXNet::NDArray|Num $other, $reverse=)
{
    confess('trying to modulo to a readonly NDArray') unless $self->writable;
    return ref $other
        ? __PACKAGE__->broadcast_mod($self, $other, { out => $self })
        : __PACKAGE__->_mod_scalar($self, $other, { out => $self })
}

=head2 empty

    Creates an empty uninitialized NDArray, with the specified shape.

    Parameters
    ----------
    $shape : Shape
        shape of the NDArray.

    :$ctx : AI::MXNet::Context, optional
        The context of the NDArray, defaults to current default context.

    :$dtype : Dtype, optional
        The dtype of the NDArray, defaults to 'float32'.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method empty(Shape $shape, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    return __PACKAGE__->new(
                handle => _new_alloc_handle(
                    $shape,
                    $ctx,
                    0,
                    DTYPE_STR_TO_MX->{$dtype}
                )
    );
}

=head2 zeros

    Creates a new NDArray filled with 0, with specified shape.

    Parameters
    ----------
    $shape : Shape
        shape of the NDArray.

    :$ctx : AI::MXNet::Context, optional
        The context of the NDArray, defaults to current default context.

    :$dtype : Dtype, optional
        The dtype of the NDArray, defaults to 'float32'.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method zeros(
    Shape $shape,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Dtype :$dtype='float32',
    Maybe[AI::MXNet::NDArray] :$out=,
    Maybe[Str] :$name=,
    Maybe[Str] :$__layout__=
)
{
    return __PACKAGE__->_zeros({ shape => $shape, ctx => "$ctx", dtype => $dtype, ($out ? (out => $out) : ())  });
}

=head2 ones

    Creates a new NDArray filled with 1, with specified shape.

    Parameters
    ----------
    $shape : Shape
        shape of the NDArray.

    :$ctx : AI::MXNet::Context, optional
        The context of the NDArray, defaults to current default context.

    :$dtype : Dtype, optional
        The dtype of the NDArray, defaults to 'float32'.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method ones(
    Shape $shape,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Dtype :$dtype='float32',
    Maybe[AI::MXNet::NDArray] :$out=,
    Maybe[Str] :$name=,
    Maybe[Str] :$__layout__=
)
{
    return __PACKAGE__->_ones({ shape => $shape, ctx => "$ctx", dtype => $dtype, ($out ? (out => $out) : ()) });
}

=head2 full

    Creates a new NDArray filled with given value, with specified shape.

    Parameters
    ----------
    $shape : Shape
        shape of the NDArray.

    val : float or int
        The value to be filled with.

    :$ctx : AI::MXNet::Context, optional
        The context of the NDArray, defaults to current default context.

    :$dtype : Dtype, optional
        The dtype of the NDArray, defaults to 'float32'.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method full(
    Shape $shape, Num $val,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Dtype :$dtype='float32', Maybe[AI::MXNet::NDArray] :$out=,
    Maybe[Str] :$name=,
    Maybe[Str] :$__layout__=
)
{
    return __PACKAGE__->_set_value({ src => $val, out => $out ? $out : __PACKAGE__->empty($shape, ctx => $ctx, dtype => $dtype) });
}

=head2 array

    Creates a new NDArray that is a copy of the source_array.

    Parameters
    ----------
    $source_array : AI::MXNet::NDArray PDL, PDL::Matrix, Array ref in PDL::pdl format
        Source data to create NDArray from.

    :$ctx : AI::MXNet::Context, optional
        The context of the NDArray, defaults to current default context.

    :$dtype : Dtype, optional
        The dtype of the NDArray, defaults to 'float32'.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method array(PDL|PDL::Matrix|ArrayRef|AI::MXNet::NDArray $source_array, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    if(blessed $source_array and $source_array->isa('AI::MXNet::NDArray'))
    {
        my $arr = __PACKAGE__->empty($source_array->shape, ctx => $ctx, dtype => $dtype);
        $arr .= $source_array;
        return $arr;
    }
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    if(not blessed($source_array))
    {
        $source_array = eval {
            pdl($pdl_type, $source_array);
        };
        confess($@) if $@;
    }
    $source_array = pdl($pdl_type, [@{ $source_array->unpdl } ? $source_array->unpdl->[0] : 0 ]) unless @{ $source_array->shape->unpdl };
    my $shape = $source_array->shape->unpdl;
    my $arr = __PACKAGE__->empty([ref($source_array) eq 'PDL' ? reverse @{ $shape } : @{ $shape }], ctx => $ctx, dtype => $dtype );
    $arr .= $source_array;
    return $arr;
}


=head2 concatenate

    Concatenates an array ref of NDArrays along the first dimension.

    Parameters
    ----------
    $arrays :  array ref of NDArrays
        Arrays to be concatenate. They must have identical shape except
        for the first dimension. They also must have the same data type.
    :$axis=0 : int
        The axis along which to concatenate.
    :$always_copy=1 : bool
        Default is 1. When not 1, if the arrays only contain one
        NDArray, that element will be returned directly, avoid copying.

    Returns
    -------
    An NDArray in the same context as $arrays->[0]->context.
=cut

method concatenate(ArrayRef[AI::MXNet::NDArray] $arrays, Index :$axis=0, :$always_copy=1)
{
    confess("no arrays provided") unless @$arrays > 0;
    if(not $always_copy and @$arrays == 1)
    {
        return $arrays->[0];
    }
    my $shape_axis = $arrays->[0]->shape->[$axis];
    my $shape_rest1 = [@{ $arrays->[0]->shape }[0..($axis-1)]];
    my $shape_rest2 = [@{ $arrays->[0]->shape }[($axis+1)..(@{ $arrays->[0]->shape }-1)]];
    my $dtype = $arrays->[0]->dtype;
    my $i = 1;
    for my $arr (@{ $arrays }[1..(@{ $arrays }-1)])
    {
        $shape_axis += $arr->shape->[$axis];
        my $arr_shape_rest1 = [@{ $arr->shape }[0..($axis-1)]];
        my $arr_shape_rest2 = [@{ $arr->shape }[($axis+1)..(@{ $arr->shape }-1)]];
        confess("first array $arrays->[0] and $i array $arr do not match")
            unless  join(',',@$arr_shape_rest1) eq join(',',@$shape_rest1);
        confess("first array $arrays->[0] and $i array $arr do not match")
            unless  join(',',@$arr_shape_rest2) eq join(',',@$shape_rest2);
        confess("first array $arrays->[0] and $i array $arr dtypes do not match")
            unless  join(',',@$arr_shape_rest2) eq join(',',@$shape_rest2);
        $i++;
    }
    my $ret_shape = [@$shape_rest1, $shape_axis, @$shape_rest2];
    my $ret = __PACKAGE__->empty($ret_shape, ctx => $arrays->[0]->context, dtype => $dtype);
    my $idx = 0;
    my $begin = [(0)x@$ret_shape];
    my $end = [@$ret_shape];
    for my $arr (@$arrays)
    {
        if ($axis == 0)
        {
            $ret->slice([$idx,($idx+$arr->shape->[0]-1)]) .= $arr;
        }
        else
        {
            $begin->[$axis] = $idx;
            $end->[$axis] = $idx+$arr->shape->[$axis];
            __PACKAGE__->_crop_assign(
                $ret, $arr,
                {
                    out => $ret,
                    begin => $begin,
                    end => $end
                }
            );
        }
        $idx += $arr->shape->[$axis];
    }
    return $ret
}

=head2 arange

    Similar function in the MXNet ndarray as numpy.arange
    See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html.

    Parameters
    ----------
    :$start=0 : number, optional
        Start of interval. The interval includes this value. The default start value is 0.
    :$stop= : number, optional
        End of interval. The interval does not include this value.
    :$step=1 : number, optional
        Spacing between the values
    :$repeat=1 : number, optional
        The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    :$ctx : Context, optional
        The context of the NDArray, defaultw to current default context.
    :$dtype : data type, optional
        The value type of the NDArray, defaults to float32

    Returns
    -------
    $out : NDArray
        The created NDArray
=cut

method arange(Index :$start=0, Maybe[Index] :$stop=, Index :$step=1, Index :$repeat=1,
              AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    return __PACKAGE__->_arange({
                start => $start,
                (defined $stop ? (stop => $stop) : ()),
                step => $step,
                repeat => $repeat,
                dtype => $dtype,
                ctx => "$ctx"
    });
}

=head2 load

    Loads ndarrays from a binary file.

    You can also use Storable to do the job if you only work with Perl.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

    Parameters
    ----------
    fname : str
        The name of the file.Can be S3 or HDFS address (remember built with S3 support).
        Example of fname:

        - `s3://my-bucket/path/my-s3-ndarray`
        - `hdfs://my-bucket/path/my-hdfs-ndarray`
        - `/path-to/my-local-ndarray`

    Returns
    -------
    $out : array ref of NDArrays or hash ref with NDArrays
=cut

method load(Str $filename)
{
    my ($handles, $names) = check_call(AI::MXNetCAPI::NDArrayLoad($filename));
    if (not @$names)
    {
        return [map { __PACKAGE__->new(handle => $_) } @$handles];
    }
    else
    {
        my $n = @$names;
        my $h = @$handles;
        confess("Handles [$h] and names [$n] count mismatch") unless $h == $n;
        my %ret;
        @ret{ @$names } = map { __PACKAGE__->new(handle => $_) } @$handles;
        return \%ret;
    }
}

=head2 save

    Save array ref of NDArray or hash of str->NDArray to a binary file.

    You can also use Storable to do the job if you only work with Perl.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

    Parameters
    ----------
    fname : str
        The name of the file.Can be S3 or HDFS address (remember built with S3 support).
        Example of fname:

        - `s3://my-bucket/path/my-s3-ndarray`
        - `hdfs://my-bucket/path/my-hdfs-ndarray`
        - `/path-to/my-local-ndarray`

    $data : array ref of NDArrays or hash ref of NDArrays
        The data to be saved.
=cut

method save(Str $filename, ArrayRef[AI::MXNet::NDArray]|HashRef[AI::MXNet::NDArray] $data)
{
    my $handles = [];
    my $names = [];
    if(ref $data eq 'HASH')
    {
        for my $name (keys %$data)
        {
            push @$names, $name;
            push @$handles, $data->{ $name }->handle;
        }
    }
    else
    {
        @$handles = map { $_->handle } @$data;
    }
    check_call(
        AI::MXNetCAPI::NDArraySave(
            $filename,
            scalar(@$handles),
            $handles,
            $names
        )
    );
}

=head2 imdecode

    Decode an image from string. Requires OpenCV to work.

    Parameters
    ----------
    $str_img : str
        binary image data
    :$clip_rect : iterable of 4 int
        clip decoded image to rectangle (x0, y0, x1, y1)
    :$out= : Maybe[NDArray]
        output buffer. can be 3 dimensional (c, h, w) or 4 dimensional (n, c, h, w)
    :$index : int
        output decoded image to i-th slice of 4 dimensional buffer
    :$channels=3 : int
        number of channels to output. Decode to grey scale when channels = 1.
    $mean= : Maybe[NDArray]
        subtract mean from decode image before outputting.
=cut

method imdecode($str_img, ArrayRef[Int] :$clip_rect=[0, 0, 0, 0],
                Maybe[AI::MXNet::NDArray] :$out=, Int :$index=0, Int :$channels=3, Maybe[AI::MXNet::NDArray] :$mean=)
{
    return __PACKAGE__->_imdecode(
        $mean//__PACKAGE__->_new_empty_handle(),
        $index,
        @$clip_rect,
        $channels,
        length($str_img),
        { str_img => $str_img, ($out ? (out => $out) : ()) }
    );
}

=head2 _new_empty_handle

    Returns a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
        a new empty ndarray handle
=cut

sub _new_empty_handle
{
    my $hdl = check_call(AI::MXNetCAPI::NDArrayCreateNone());
    return $hdl;
}

=head2 _new_alloc_handle

    Returns a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    a new empty ndarray handle
=cut

func _new_alloc_handle($shape, $ctx, $delay_alloc, $dtype)
{
    my $hdl = check_call(AI::MXNetCAPI::NDArrayCreateEx(
        $shape,
        scalar(@$shape),
        $ctx->device_type_id,
        $ctx->device_id,
        $delay_alloc,
        $dtype)
    );
    return $hdl;
}

=head2 waitall

    Wait for all async operations to finish in MXNet.
    This function is used for benchmarks only.
=cut

method waitall()
{
    check_call(AI::MXNetCAPI::NDArrayWaitAll());
}

=head2 _fresh_grad

        Parameters:
        ----------
        Maybe[Bool] $state=

        Whether this array's corresponding gradient array
        (registered via `autograd->mark_variables`) has been
        updated by `autograd->backward` since last reset.

        `_fresh_grad` need to be manually set to False
        after consuming gradient (usually after updating this
        array).
=cut

method _fresh_grad(Maybe[Bool] $state=)
{
    if(defined $state)
    {
        check_call(AI::MXNetCAPI::NDArraySetGradState($self->handle, $state));
        return $state;
    }
    else
    {
        return scalar(check_call(AI::MXNetCAPI::NDArrayGetGradState($self->handle)));
    }
}

=head2 detach

    Returns a new NDArray, detached from the current graph.
=cut

method detach()
{
    my $handle = check_call(AI::MXNetCAPI::NDArrayDetach($self->handle));
    return __PACKAGE__->new(handle => $handle);
}

=head2 attach_grad

        Attach a gradient buffer to this NDArray, so that `backward`
        can compute gradient with respect to it.

        Parameters
        ----------
        GradReq :$grad_req='write' : {'write', 'add', 'null'}
            How gradient will be accumulated.
            - 'write': gradient will be overwritten on every backward.
            - 'add': gradient will be added to existing value on every backward.
            - 'null': do not compute gradient for this NDArray.
        Maybe[Str] :$stype= : str, optional
            The storage type of the gradient array. Defaults to the same stype of this NDArray.
=cut

method attach_grad(GradReq :$grad_req='write', Maybe[Str] :$stype=)
{
    my $grad;
    if(defined $stype)
    {
        $grad = __PACKAGE__->_zeros($self->shape, stype=>$stype);
    }
    else
    {
        $grad = $self->zeros_like;
    }
    $grad_req = GRAD_REQ_MAP->{$grad_req};
    check_call(
        AI::MXNetCAPI::AutogradMarkVariables(
            1,
            [$self->handle],
            [$grad_req],
            [$grad->handle]
        )
    );
}

=head2 grad

    Returns gradient buffer attached to this NDArray.
=cut

method grad()
{
    my $handle = check_call(AI::MXNetCAPI::NDArrayGetGrad($self->handle));
    return undef unless defined $handle;
    return __PACKAGE__->new(handle => $handle);
}

=head2 backward

    Compute the gradients of this NDArray w.r.t variables.

    Parameters
    ----------
    :$out_grad= : NDArray, optional
        Gradient with respect to head.
    :$retain_graph=0 : bool, optional
        Whether to retain the computaion graph for another backward
        pass on the same graph. By default the computaion history
        is cleared.
    :$train_mode=1 : bool, optional
        Whether to compute gradient for training or inference.
=cut

method backward(Maybe[AI::MXNet::NDArray] :$out_grad=, Bool :$retain_graph=0, Bool :$train_mode=1)
{
    check_call(
        AI::MXNetCAPI::AutogradBackwardEx(
            1,
            [$self->handle],
            [defined $out_grad ? $out_grad->handle : undef],
            0,
            [],
            $retain_graph,
            0,
            $train_mode
        )
    )
}

method CachedOp(@args) { AI::MXNet::CachedOp->new(@args) }

my $lvalue_methods = join "\n", map {"use attributes 'AI::MXNet::NDArray', \\&AI::MXNet::NDArray::$_, 'lvalue';"}
qw/at slice aspdl asmpdl reshape copy sever T astype as_in_context copyto empty zero ones full
                       array/;
eval << "EOV" if ($^V and $^V >= 5.006007);
{
  no warnings qw(misc);
  $lvalue_methods
}
EOV

sub contrib { 'AI::MXNet::Contrib::NDArray' }
sub random  { 'AI::MXNet::Random' }

__PACKAGE__->meta->make_immutable;
