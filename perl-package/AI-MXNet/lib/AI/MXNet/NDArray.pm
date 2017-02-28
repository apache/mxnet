package AI::MXNet::NDArray;
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
    '**' => \&power,
    '==' => \&equal,
    '!=' => \&not_equal,
    '>'  => \&greater,
    '>=' => \&greater_equal,
    '<'  => \&lesser,
    '<=' => \&lesser_equal,
    '.=' => \&set,
    '=' => sub { $_[0] };

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
    zip(sub {
        my ($idx, $dim_size) = @_;
        confess("Dimension $i mismatch Idx: $idx >= Dim Size: $dim_size")
            if $idx >= $dim_size or ($idx + $dim_size) < 0;
        ++$i;
    }, \@indices, $shape);  
    $i = 0;
    for my $v (@indices)
    {
        $v += $shape->[$i] if $v < 0;
        ++$i;
    }
    return $self->_at($indices[0]) if @indices == 1;
    return $self->slice(@indices);
}

method slice(Slice @slices)
{
    confess("No slices supplied") unless @slices;
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
        ref $_ ? (@$_ == 1 ? [$_->[0], $shape->[$i] - 1] : $_) : ($_ eq 'X' ? [0, $shape->[$i] - 1] : [$_, $_]); 
    } @slices; 
    zip(sub {
        my ($slice, $dim_size) = @_;
        my ($begin, $end, $stride) = @$slice;
        confess("NDArray does not support slice strides != 1")
            if ($stride//0) > 1;
        confess("Dimension $i mismatch slice begin : $begin >= Dim Size: $dim_size")
            if $begin >= $dim_size or ($begin + $dim_size) < 0;
        confess("Dimension $i mismatch slice end : $end >= Dim Size: $dim_size")
            if $end >= $dim_size or ($end + $dim_size) < 0;
    }, \@slices, $shape);  
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
        $self->_set_value($value, { out => $self });
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
}

=head2 _sync_copyfrom(self, source_array)

        Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        Can be array ref in PDL::pdl init format or PDL object
=cut

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
    my $buf;
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $buf = pack("S*", map { AI::MXNetCAPI::_float_to_half($_) } unpack ("f*", ${$source_array->get_dataref}));
    }
    else
    {
        $buf = ${$source_array->get_dataref};
    }
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromCPU($self->handle, $buf, $self->size));
    return $self;
}

=head2 aspdl

        Return a copied PDL array of current array.

        Returns
        -------
        array : PDL
        A copy of array content.
=cut

method aspdl()
{
    my $dtype = $self->dtype;
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    my $pdl = PDL->new_from_specification($pdl_type, reverse @{ $self->shape });
    my $perl_pack_type = DTYPE_MX_TO_PERL->{$dtype};
    my $buf = pack("$perl_pack_type*", (0)x$self->size);
    check_call(AI::MXNetCAPI::NDArraySyncCopyToCPU($self->handle, $buf, $self->size)); 
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $buf = pack("f*", map { AI::MXNetCAPI::_half_to_float($_) } unpack("S*", $buf));
    }
    ${$pdl->get_dataref} = $buf;
    $pdl->upd_data;
    return $pdl;
}


=head2 asmpdl

        Return a copied PDL::Matrix array of current array.

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
    my $buf = pack("$perl_pack_type*", (0)x$self->size);
    check_call(AI::MXNetCAPI::NDArraySyncCopyToCPU($self->handle, $buf, $self->size)); 
    ## special handling for float16
    if($perl_pack_type eq 'S')
    {
        $buf = pack("f*", map { AI::MXNetCAPI::_half_to_float($_) } unpack("S*", $buf));
    }
    ${$pdl->get_dataref} = $buf;
    $pdl->upd_data;
    return $pdl;
}


=head2 _slice

        Return a sliced NDArray that shares memory with current one.

        Parameters
        ----------
        start : int
            Starting index of slice.
        stop : int
            Finishing index of slice.
=cut

method  _slice (
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

        Return a sub NDArray that shares memory with current one.

        Parameters
        ----------
        idx : int
            index of sub array.
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
 
        Return a reshaped NDArray that shares memory with current one.

        Parameters
        ----------
        new_shape : iterable of int
            new shape of NDArray
=cut

method reshape(Shape $new_shape)
{
    confess("new size does not mach old size")
        unless __PACKAGE__->size($new_shape) == $self->size;
    my $handle = check_call(
                    AI::MXNetCAPI::NDArrayReshape(
                        $self->handle,
                        scalar(@$new_shape),
                        $new_shape
                    )
    );
    return __PACKAGE__->new(handle => $handle, writable => $self->writable);
}


=head broadcast_to

        Broadcasting the current NDArray into the given shape. The semantics is
        the same with `numpy`'s broadcasting

        Parameters
        ---------
        shape : the shape to broadcast
            the broadcast shape
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

        Block until all pending writes operations on current NDArray are finished.

        This function will return when all the pending writes to the current
        NDArray finishes. There can still be pending read going on when the
        function returns.
=cut

method wait_to_read()
{
    check_call(AI::MXNetCAPI::NDArrayWaitToRead($self->handle));
}

=head2 shape

        Get shape of current NDArray.

        Returns
        -------
        a tuple representing shape of current ndarray
=cut

method shape()
{
    return scalar(check_call(AI::MXNetCAPI::NDArrayGetShape($self->handle)));
}

=head2 size

        Get size of current NDArray.

        Returns
        -------
        an int representing size of current ndarray
=cut

method size(Shape|Undef $shape=)
{
    my $size = 1;
    map { $size *= $_ } @{ $shape//$self->shape };
    return $size;
}


=head2 context

        Get context of current NDArray.

        Returns
        -------
        context : mxnet.Context
            The context of current NDArray.
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

        Get data type of current NDArray.

        Returns
        -------
        an dtype string ('float32', 'float64', 'float16', 'uint8', 'int32') 
        representing type of current ndarray.
        'float32' is a default dtype for ndarray class.
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
        Copy the content of current array to other.

        When other is NDArray, the content is copied over.
        When other is a Context, a new NDArray in the context
        will be created as target

        Parameters
        ----------
        other : NDArray or Context
            Target NDArray or context we want to copy data to.

        Returns
        -------
        dst : NDArray
            The copy target NDArray
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

        Make a copy of the current ndarray on the same context

        Return
        ------
        cpy : NDArray
            The copy
=cut

method copy()
{
    return $self->copyto($self->context);
}

## alias for PDL::NiceSlice
*sever = \&copy;

=head2 T

        Get transpose of current NDArray
=cut

method T()
{
    if (@{$self->shape} != 2)
    {
        confess('Only 2D matrix is allowed to be transposed');
    }
    return __PACKAGE__->transpose($self);
}

=head2 astype
        Return a copied ndarray of current array with specified type.

        Parameters
        ----------
        dtype : numpy.dtype or string
            Desired type of result array.

        Returns
        -------
        array : ndarray
            A copy of array content.
=cut

method astype(Dtype $dtype)
{
    my $res = __PACKAGE__->empty($self->shape, ctx => $self->context, dtype => $dtype);
    $self->copyto($res);
    return $res;
}

=head2 as_in_context

        Return an `NDArray` that lives in the target context. If the array
        is already in that context, `self` is returned. Otherwise, a copy is
        made.

        Parameters
        ----------
        context : Context
            The target context we want the return value to live in.

        Returns
        -------
        A copy or `self` as an `NDArray` that lives in the target context.
=cut

method as_in_context(AI::MXNet::Context $context)
{
    return $self if $self->context == $context;
    return $self->copyto($context);
}

=head onehot_encode

    One hot encoding indices into matrix out.

    Parameters
    ----------
    indices: NDArray
        An NDArray containing indices of the categorical features.

    out: NDArray
        The result holder of the encoding.

    Returns
    -------
    out: Array
        Same as out.
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
        left hande side operand

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

=head2 add

    Perform element-wise addition

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method add(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_add _plus_scalar/
    );
}

=head2 subtract

    Perform element-wise subtract

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

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


=head2 multiply

    Perform element-wise multiplication

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

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


=head2 divide

    Perform element-wise divide

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

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

=head2 power

    Perform element-wise power operator

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method power(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_power _power_scalar _rpower_scalar/,
        $reverse
    );
}

=head2 maximum

    Perform maximum operator

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    Returns
    -------
    out: Array
        result array
=cut

method maximum(AI::MXNet::NDArray|Num $other)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_maximum _maximum_scalar/
    );
}

=head2 minimum

    Perform minimum operator

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    Returns
    -------
    out: Array
        result array
=cut

method minimum(AI::MXNet::NDArray|Num $other)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_minimum _minimum_scalar/
    );
}

=head2 equal

    Return ($self == $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_equal _equal_scalar/
    );
}

=head2 not_equal

    Return ($self != $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method not_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_not_equal _not_equal_scalar/
    );
}

=head2 greater

    Return ($self > $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method greater(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_greater _greater_scalar _lesser_scalar/,
        $reverse
    );
}

=head2 greater_equal

    Return ($self >= $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method greater_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_greater_equal _greater_equal_scalar _lesser_equal_scalar/,
        $reverse
    );
}

=head2 lesser

    Return ($self < $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method lesser(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_lesser _lesser_scalar _greater_scalar/,
        $reverse
    );
}

=head2 lesser_equal

    Return ($self <= $other) element-wise

    Parameters
    ----------
    $other : Array of float value
        right hand side operand
    $reverse : Boolean, if true,
        reverses $self and $other
    Returns
    -------
    out: Array
        result array
=cut

method lesser_equal(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return _ufunc_helper(
        $self,
        $other,
        qw/broadcast_lesser_equal _lesser_equal_scalar _greater_equal_scalar/,
        $reverse
    );
}

=head2 true_divide

    The same as divide
=cut

method true_divide(AI::MXNet::NDArray|Num $other, $reverse=)
{
    return $self->divide($other, $reverse);
}

=head2 empty(

    Create an empty uninitialized new NDArray, with specified shape.

    Parameters
    ----------
    shape : ArrayRef
        shape of the NDArray.

    ctx : Context, optional
        The context of the NDArray, default to current default context.

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

    Create a new NDArray filled with 0, with specified shape.

    Parameters
    ----------
    shape : ArrayRef
        shape of the NDArray.
    ctx : Context, optional.
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method zeros(Shape $shape, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    return __PACKAGE__->_zeros({ shape => $shape, ctx => "$ctx", dtype => $dtype });
}

=head2 ones

    Create a new NDArray filled with 1, with specified shape.

    Parameters
    ----------
    shape : ArrayRef
        shape of the NDArray.
    ctx : Context, optional.
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method ones(Shape $shape, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    return __PACKAGE__->_ones({ shape => $shape, ctx => "$ctx", dtype => $dtype });
}

=head2 full

    Create a new NDArray filled with given value, with specified shape.

    Parameters
    ----------
    shape : ArrayRef
        shape of the NDArray.
    val : float or int
        value to be filled with.
    ctx : Context, optional.
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method full(Shape $shape, Num $val, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
    return __PACKAGE__->_set_value({ src => $val, out => __PACKAGE__->empty($shape, ctx => $ctx, dtype => $dtype) });
}

=head2 array

    Create a new NDArray that copies content from source_array.

    Parameters
    ----------
    source_array : array_like
        Source data to create NDArray from.

    ctx : Context, optional
        The context of the NDArray, default to current default context.

    Returns
    -------
    out: Array
        The created NDArray.
=cut

method array(PDL|PDL::Matrix|ArrayRef $source_array, AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx, Dtype :$dtype='float32')
{
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

    Concatenate a list of NDArrays along the first dimension.

    Parameters
    ----------
    arrays : list of NDArray
        Arrays to be concatenate. They must have identical shape except
        the first dimension. They also must have the same data type.
    axis : int
        The axis along which to concatenate.
    always_copy : bool
        Default `True`. When not `True`, if the arrays only contain one
        `NDArray`, that element will be returned directly, avoid copying.

    Returns
    -------
    An `NDArray` that lives on the same context as `arrays[0].context`.
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

    Simlar function in the MXNet ndarray as numpy.arange
        See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value. The default start value is 0.
    stop : number, optional
        End of interval. The interval does not include this value.
    step : number, optional
        Spacing between values
    repeat : number, optional
        "The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    ctx : Context, optional
        The context of the NDArray, default to current default context.
    dtype : type, optional
        The value type of the NDArray, default to np.float32

    Returns
    -------
    out : NDArray
        The created NDArray
=cut

method arange(Index :$start=0, Index :$stop=, Index :$step=1, Index :$repeat=1,
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

    Load ndarray from binary file.

    You can also use Storable to do the job if you only work on perl.
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
    out : list of NDArray or dict of str to NDArray
        List of NDArray or dict of str->NDArray, depending on what was saved.
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

    Save array of NDArray or hash of str->NDArray to binary file.

    You can also use Storable to do the job if you only work on perl.
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

    data : list of NDArray or dict of str to NDArray
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
    str_img : str
        binary image data
    clip_rect : iterable of 4 int
        clip decoded image to rectangle (x0, y0, x1, y1)
    out : NDArray
        output buffer. can be 3 dimensional (c, h, w) or 4 dimensional (n, c, h, w)
    index : int
        output decoded image to i-th slice of 4 dimensional buffer
    channels : int
        number of channels to output. Decode to grey scale when channels = 1.
    mean : NDArray
        subtract mean from decode image before outputing.
=cut

method imdecode($str_img, ArrayRef[Int] :$clip_rect=[0, 0, 0, 0],
                AI::MXNet::NDArray :$out=, Int :$index=0, Int :$channels=3, AI::MXNet::NDArray :$mean=)
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

    """Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty ndarray handle
    """
=cut

sub _new_empty_handle
{
    my $hdl = check_call(AI::MXNetCAPI::NDArrayCreateNone());
    return $hdl;
}

=head2 _new_alloc_handle

    Return a new handle with specified shape and context.

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

=head2
    Wait all async operation to finish in MXNet

    This function is used for benchmark only
    """
=cut

method waitall()
{
    check_call(AI::MXNetCAPI::NDArrayWaitAll());
}

my $lvalue_methods = join "\n", map {"use attributes 'AI::MXNet::NDArray', \\&AI::MXNet::NDArray::$_, 'lvalue';"}
qw/at slice aspdl asmpdl reshape copy sever T astype as_in_context copyto empty zero ones full
                       array/;
eval << "EOV" if ($^V and $^V >= 5.006007);
{ 
  no warnings qw(misc);
  $lvalue_methods
}
EOV

__PACKAGE__->meta->make_immutable;
