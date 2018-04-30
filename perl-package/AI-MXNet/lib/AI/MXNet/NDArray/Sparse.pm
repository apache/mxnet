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

package AI::MXNet::NDArray::Sparse;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Mouse;
extends 'AI::MXNet::NDArray';

=head1 NAME

    AI::MXNet::NDArray::Sparse - Sparse NDArray API of MXNet
=cut

=head1 DESCRIPTION

    The base class of an NDArray stored in a sparse storage format.
    See AI::MXNet::NDArray::CSR and AI::MXNet::NDArray::RowSparse for more details.
=cut

method _new_alloc_handle(
    Stype                    $stype,
    Shape                    $shape,
    AI::MXNet::Context       $ctx,
    Bool                     $delay_alloc,
    Dtype                    $dtype,
    AuxTypes                 $aux_types,
    Maybe[ArrayRef[Shape]]   $aux_shapes=
)
{
    confess("only int64 is supported for aux types")
        if (grep { $_ ne 'int64' } @$aux_types);
    my $aux_type_ids = [map { DTYPE_STR_TO_MX->{$_} } @$aux_types];
    $aux_shapes //= [map { [0] } @$aux_types];
    my $aux_shape_lens = [map { scalar(@$_) } @$aux_shapes];
    @$aux_shapes = map { @$_ } @$aux_shapes;
    my $num_aux = @{ $aux_types };
    my $handle = check_call(
        AI::MXNetCAPI::NDArrayCreateSparseEx(
            STORAGE_TYPE_STR_TO_ID->{$stype},
            $shape,
            scalar(@$shape),
            $ctx->device_type_id,
            $ctx->device_id,
            $delay_alloc,
            DTYPE_STR_TO_MX->{$dtype},
            scalar(@$aux_types),
            $aux_type_ids,
            $aux_shape_lens,
            $aux_shapes
        )
    );
}

method _class_name()
{
    my $class = ref $self || $self;
    $class;
}

sub not_implemented { confess "Not implemented" }
use overload '""' => sub {
                        my $self = shift;
                        my $shape_info = join('x', @{ $self->shape });
                        sprintf("\n<%s, %s @%s>", $self->_class_name, $shape_info, $self->context);
                     },
             '+'  => \&add,
             '-'  => \&subtract,
             '*'  => \&multiply,
             '/'  => \&divide,
             '+=' => \&not_implemented,
             '-=' => \&not_implemented,
             '*=' => \&not_implemented,
             '/=' => \&not_implemented;

method add(AI::MXNet::NDArray|Num $other, $reverse=)
{
    if(blessed $other and join(',', @{ $self->shape }) eq join(',', @{ $other->shape }))
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/elemwise_add _plus_scalar/
        );
    }
    else
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/broadcast_add _plus_scalar/
        );
    }
}


method subtract(AI::MXNet::NDArray|Num $other, $reverse=)
{
    if(blessed $other and join(',', @{ $self->shape }) eq join(',', @{ $other->shape }))
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/elemwise_sub _minus_scalar _rminus_scalar/,
            $reverse
        );
    }
    else
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/broadcast_sub _minus_scalar _rminus_scalar/,
            $reverse
        );
    }
}

method multiply(AI::MXNet::NDArray|Num $other, $reverse=)
{
    if(blessed $other and join(',', @{ $self->shape }) eq join(',', @{ $other->shape }))
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/elemwise_mul _mul_scalar/,
        );
    }
    else
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/broadcast_mul _mul_scalar/,
        );
    }
}

method divide(AI::MXNet::NDArray|Num $other, $reverse=)
{
    if(blessed $other and join(',', @{ $self->shape }) eq join(',', @{ $other->shape }))
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/elemwise_div _div_scalar _rdiv_scalar/,
            $reverse
        );
    }
    else
    {
        return AI::MXNet::NDArray::_ufunc_helper(
            $self,
            $other,
            qw/broadcast_div _div_scalar _rdiv_scalar/,
            $reverse
        );
    }
}

{
    no warnings 'redefine';
    *_sync_copyfrom = *_at = *_slice = *reshape = *size = \&not_implemented;
}

method _aux_type(Int $i)
{
    return DTYPE_MX_TO_STR->{
        check_call(
            AI::MXNetCAPI::NDArrayGetAuxType(
                $self->handle, $i
            )
        )
    }
}

method _num_aux()
{
    return scalar(@{ STORAGE_AUX_TYPES->{ $self->stype } });
}

method _aux_types()
{
    [map { $self->_aux_type($_) } 0..$self->_num_aux-1];
}

=head2 aspdl

    Return a dense PDL object with value copied from this array
=cut

method aspdl()
{
    return $self->tostype('default')->aspdl;
}

=head2 astype

        Returns a copy of the array after casting to a specified type.
        Parameters
        ----------
        dtype : Dtype
            The type of the returned array.
        Examples
        --------
        >>> $x = mx->nd->sparse->zeros('row_sparse', [2,3], dtype=>'float32')
        >>> $y = $x->astype('int32')
        >>> $y->dtype
        <type 'int32'>
=cut

method astype(Dtype $dtype)
{
    my $res = $self->zeros(
        $self->stype, $self->shape, ctx => $self->context,
        dtype => $dtype
    );
    $self->copyto($res);
    return $res;
}

=head2 copyto

        Copies the value of this array to another array.

        Parameters
        ----------
        other : NDArray or NDArray::CSR or NDArray::RowSparse or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray::CSR or NDArray::RowSparse
            The copied array.
=cut

method copyto(AI::MXNet::NDArray|AI::MXNet::Context $other)
{
    if($other->isa('AI::MXNet::NDArray'))
    {
        if($self->handle eq $other->handle)
        {
            Carp::cluck('You are attempting to copy an array to itself');
            return;
        }
        else
        {
            return __PACKAGE__->_copyto($self, out => $other);
        }
    }
    elsif($other->isa('AI::MXNet::Context'))
    {
        my $hret = __PACKAGE__->_ndarray_cls(
            __PACKAGE__->_new_alloc_handle(
                $self->stype, $self->shape, $other, 1, $self->dtype, $self->_aux_types
            )
        );
        return __PACKAGE__->_copyto($self, out=>$hret)
    }
}

=head2 check_format

        Check whether the NDArray format is valid.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
=cut

method check_format(Bool $full_check=1)
{
    scalar(check_call(AI::MXNetCAPI::NDArraySyncCheckFormat($self->handle, $full_check)));
}

=head2 _data

        A deep copy NDArray of the data array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
=cut

method _data()
{
    $self->wait_to_read;
    my $handle = check_call(AI::MXNetCAPI::NDArrayGetDataNDArray($self->handle));
    return AI::MXNet::NDArray->new(handle => $handle);
}

=head2 _aux_data

        Get a deep copy NDArray of the i-th aux data array associated with the
        AI::MXNet::NDArray::Sparse

        This function blocks. Do not use it in performance critical code.
=cut

method _aux_data(Int $i)
{
    $self->wait_to_read;
    my $handle = check_call(AI::MXNetCAPI::NDArrayGetAuxNDArray($self->handle, $i));
    return AI::MXNet::NDArray->new(handle => $handle);
}

package AI::MXNet::NDArray::CSR;
use AI::MXNet::Base;
use Mouse;
extends 'AI::MXNet::NDArray::Sparse';

=head1 NAME

    AI::MXNet::NDArray::CSR - A sparse representation of 2D NDArray in the Compressed Sparse Row format.
=cut

=head1 DESCRIPTION

    A AI::MXNet::NDArray::CSR represents an AI::MXNet::NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the CSR representation where the column indices for
    row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values are stored
    in ``data[indptr[i]:indptr[i+1]]``.

    The column indices for a given row are expected to be sorted in ascending order.
    Duplicate column entries for the same row are not allowed.

    Example
    -------
    >>> $a = mx->nd->array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]]);
    >>> $a = $a->tostype('csr');
    >>> $a->data->aspdl;
    [ 1  2  3]
    >>> $a->indices->aspdl
    [1 0 2]
    >>> $a->indptr->aspdl
    [0 1 2 2 3]

    See Also
    --------
    csr_matrix: Several ways to construct a CSRNDArray
=cut

#    def __reduce__(self):
#        return CSRNDArray, (None,), super(CSRNDArray, self).__getstate__()

use overload '+=' => sub { ($_[0] + $_[1])->copyto($_[0]) },
             '-=' => sub { ($_[0] - $_[1])->copyto($_[0]) },
             '*=' => sub { ($_[0] * $_[1])->copyto($_[0]) },
             '/=' => sub { ($_[0] / $_[1])->copyto($_[0]) };

=head2 slice

        Returns a newly created array based on the indexing key.

        Parameters
        ----------
        key : int or array ref
            Indexing key.

        Examples
        --------
        >>> $indptr = [0, 2, 3, 6];
        >>> $indices = [0, 2, 2, 0, 1, 2];
        >>> $data = [1, 2, 3, 4, 5, 6];
        >>> $a = mx->nd->sparse->csr_matrix([$data, $indices, $indptr], shape=>[3, 3])
        >>> $a->aspdl
            [[ 1  0  2]
             [ 0  0  3]
             [ 4  5  6]]
        >>> $a->slice([1,2])->aspdl
        [[ 0  0  3]]
        >>> $a->slice(1)->aspdl
        [[ 0  0  3]]
        >>> $a->[-1]->aspdl
        [[ 4  5  6]]
=cut

method slice(Slice|InternalSlice @slices)
{
    if(grep { /^begin|end|slice$/ } @slices)
    {
        return $self->SUPER::slice(@slices);
    }
    my $slice = $slices[0];
    my ($begin, $end);
    if(not ref $slice)
    {
        if($slice < 0)
        {
            $begin = $self->shape->[0] + $slice;
        }
        else
        {
            $begin = $slice;
        }
        $end = $begin;
    }
    else
    {
        ($begin, $end) = @{ $slice };
        $end //= $self->shape->[0] - 1;
        if($begin < 0)
        {
            $begin += $self->shape->[0];
        }
        if($end < 0)
        {
            $end += $self->shape->[0];
        }
    }
    return $self->SUPER::slice(begin => $begin, end => $end + 1);
}


=head2 set

        Set self to value. Also usable as overloaded .=

        Parameters
        ----------
        value : AI::MXNet::NDArray or AI::MXNet::NDArray::CSR
                or PDL/PDL::CCS::Nd/perl array ref in PDL constructor format
            The value to set.

        Examples
        --------
        >>> $src = mx->nd->sparse->zeros('csr', [3,3])
        >>> $src->aspdl
              [[ 0  0  0]
               [ 0  0  0]
               [ 0  0  0]]
        >>> # AI::MXNet::NDArray::CSR with same storage type
        >>> $x = mx->nd->ones('row_sparse', [3,3])->tostype('csr')
        >>> $x .= $src
        >>> $x->aspdl
              [[ 1  1  1]
               [ 1  1  1]
               [ 1  1  1]]
        >>> # assign NDArray to AI::MXNet::NDArray::CSR
        >>> $x .= mx->nd->ones([3,3]) * 2
        >>> $x->aspdl
              [[ 2  2  2]
               [ 2  2  2]
               [ 2  2  2]]
=cut

method set(AcceptableInput $other, $reverse=)
{
    confess('Failed to assign to a readonly CSR') unless $self->writable;
    if($other->isa('AI::MXNet::NDArray'))
    {
        if($self->handle ne $other->handle)
        {
            $other->copyto($self);
        }
    }
    else
    {
        my $tmp = __PACKAGE__->array($other, ((not ref $other) ? ( pdl => $self->aspdl) : ()));
        $tmp->copyto($self);
    }
}

use overload '.=' => \&set;

=head2 indices

        A deep copy NDArray of the indices array of the AI::MXNet::NDArray::CSR.
        This generates a deep copy of the column indices of the current `csr` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::CSR indices array.
=cut

method indices()
{
    return $self->_aux_data(1);
}

=head2 indptr

        A deep copy NDArray of the inptr array of the AI::MXNet::NDArray::CSR.
        This generates a deep copy of the indptr of the current `csr` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::CSR indptr array.
=cut

method indptr()
{
    return $self->_aux_data(0);
}

=head2 data

        A deep copy NDArray of the data array of the AI::MXNet::NDArray::CSR.
        This generates a deep copy of the data of the current `csr` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::CSR data array.
=cut

method data()
{
    return $self->_data;
}

=head2 tostype

        Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or AI::MXNet::NDArray::CSR 
            A copy of the array with the chosen storage stype
=cut

method tostype(Stype $stype)
{
    if($stype eq 'row_sparse')
    {
        confess("cast_storage from csr to row_sparse is not supported");
    }
    return $self->cast_storage(stype => $stype);
}

=head2 copyto

        Copies the value of this array to another array.

        If $other is a AI::MXNet::NDArray or AI::MXNet::NDArray::CSR object, then $other->shape and
        $self->shape should be the same. This function copies the value from
        $self to $other.

        If $other is a context, a new AI::MXNet::NDArray::CSR will be first created on
        the target context, and the value of $self is copied.

        Parameters
        ----------
        $other : AI::MXNet::NDArray or AI::MXNet::NDArray::CSR or AI::MXNet::Context
            The destination array or context.

        Returns
        -------
        AI::MXNet::NDArray or AI::MXNet::NDArray::CSR
=cut

method copyto(AI::MXNet::Context|AI::MXNet::NDArray $other)
{
    if($other->isa('AI::MXNet::Context'))
    {
        return $self->SUPER::copyto($other);
    }
    else
    {
        my $stype = $other->stype;
        if($stype eq 'default' or $stype eq 'csr')
        {
            return return $self->SUPER::copyto($other);
        }
        else
        {
            confess("copyto does not support destination NDArray stype $stype");
        }
    }
}

=head2

    Returns a PDL::CCS::Nd object with value copied from this array
=cut

method aspdlccs()
{
    return ascsr($self->data->aspdl, $self->indptr->aspdl, $self->indices->aspdl, $self->shape);
}

package AI::MXNet::NDArray::RowSparse;
use Mouse;
extends 'AI::MXNet::NDArray::Sparse';

=head1 NAME

    AI::MXNet::NDArray::RowSparse - A sparse representation of a set of NDArray row slices at given indices.
=cut

=head1 DESCRIPTION

    A AI::MXNet::NDArray::RowSparse represents a multidimensional NDArray using two separate arrays: `data` and
    `indices`. The number of dimensions has to be at least 2.

    - data: an NDArray of any dtype with shape [D0, D1, ..., Dn].
    - indices: a 1-D int64 NDArray with shape [D0] with values sorted in ascending order.

    The `indices` stores the indices of the row slices with non-zeros,
    while the values are stored in `data`. The corresponding NDArray ``dense``
    represented by AI::MXNet::NDArray::RowSparse ``rsp`` has

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

        >>> $dense->aspdl
              [[ 1  2  3 ]
               [ 0  0  0 ]
               [ 4  0  5 ]
               [ 0  0  0 ]
               [ 0  0  0 ]]
        >>> $rsp = $dense->tostype('row_sparse');
        >>> $rsp->indices->aspdl
              [ 0 2 ]
        >>> $rsp->data->aspdl
              [[ 1  2 3 ]
               [ 4  0 5 ]]

    A AI::MXNet::NDArray::RowSparse is typically used to represent non-zero row slices of a large NDArray
    of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

    AI::MXNet::NDArray::RowSparse is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. sparse dot and sparse embedding).

    See Also
    --------
    row_sparse_array: Several ways to construct a AI::MXNet::NDArray::RowSparse
=cut

use overload '+=' => sub { ($_[0] + $_[1])->copyto($_[0]) },
             '-=' => sub { ($_[0] - $_[1])->copyto($_[0]) },
             '*=' => sub { ($_[0] * $_[1])->copyto($_[0]) },
             '/=' => sub { ($_[0] / $_[1])->copyto($_[0]) };

method slice(@args) { confess("not implemented") }

=head2 set

        Set self to value. Also usable as overloaded .=

        Parameters
        ----------
        value : AI::MXNet::NDArray or AI::MXNet::NDArray::CSR
                or PDL/PDL::CCS::Nd/perl array ref in PDL constructor format
            The value to set.

        Examples
        --------
        >>> $src = mx->nd->sparse->zeros('raw_sparse', [3,3])
        >>> $src->aspdl
              [[ 0  0  0]
               [ 0  0  0]
               [ 0  0  0]]
        >>> # AI::MXNet::NDArray::RowSparse with same storage type
        >>> $x = mx->nd->ones('row_sparse', [3,3])
        >>> $src .= $x
        >>> $src->aspdl
              [[ 1  1  1]
               [ 1  1  1]
               [ 1  1  1]]
        >>> # assign NDArray to AI::MXNet::NDArray::RowSparse
        >>> $x .= mx->nd->ones([3,3]) * 2
        >>> $x->aspdl
              [[ 2  2  2]
               [ 2  2  2]
               [ 2  2  2]]
=cut

method set(AcceptableInput $other, $reverse=)
{
    confess('Failed to assign to a readonly RowSparse') unless $self->writable;
    if($other->isa('AI::MXNet::NDArray'))
    {
        if($self->handle ne $other->handle)
        {
            $other->copyto($self);
        }
    }
    else
    {
        my $tmp = __PACKAGE__->array($other, ((not ref $other) ? ( pdl => $self->aspdl) : ()));
        $tmp->copyto($self);
    }
}

use overload '.=' => \&set;

=head2 data

        A deep copy NDArray of the data array of the AI::MXNet::NDArray::RowSparse.
        This generates a deep copy of the data of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::RowSparse data array.
=cut

method data()
{
    return $self->_data;
}

=head2 indices

        A deep copy NDArray of the indices array of the AI::MXNet::NDArray::RowSparse.
        This generates a deep copy of the column indices of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::RowSparse indices array.
=cut

method indices()
{
    return $self->_aux_data(0);
}

=head2 data

        A deep copy NDArray of the data array of the AI::MXNet::NDArray::RowSparse.
        This generates a deep copy of the data of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This AI::MXNet::NDArray::RowSparse data array.
=cut

=head2 tostype

        Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or RowSparseNDArray
            A copy of the array with the chosen storage stype
=cut

method tostype(Stype $stype)
{
    if($stype eq 'csr')
    {
        confess("cast_storage from row_sparse to csr is not supported");
    }
    return $self->cast_storage(stype => $stype);
}


=head2 copyto

        Copies the value of this array to another array.

        If $other is a AI::MXNet::NDArray or AI::MXNet::NDArray::RawSparse object, then $other->shape and
        $self->shape should be the same. This function copies the value from
        $self to $other.

        If $other is a context, a new AI::MXNet::NDArray::RawSparse will be first created on
        the target context, and the value of $self is copied.

        Parameters
        ----------
        $other : AI::MXNet::NDArray or AI::MXNet::NDArray::RawSparse or AI::MXNet::Context
            The destination array or context.

        Returns
        -------
        AI::MXNet::NDArray or AI::MXNet::NDArray::RawSparse
=cut

method copyto(AI::MXNet::Context|AI::MXNet::NDArray $other)
{
    if($other->isa('AI::MXNet::Context'))
    {
        return $self->SUPER::copyto($other);
    }
    else
    {
        my $stype = $other->stype;
        if($stype eq 'default' or $stype eq 'row_sparse')
        {
            return return $self->SUPER::copyto($other);
        }
        else
        {
            confess("copyto does not support destination NDArray stype $stype");
        }
    }
}

package AI::MXNet::NDArray::Sparse;

# Prepare `source_array` so that it can be used to construct NDArray.
# `source_array` is converted to a `pdl` if it's neither an `NDArray`
# nor a `pdl`.

method _prepare_src_array($source_array, Dtype $dtype)
{
    my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype });
    if(not blessed($source_array))
    {
        $source_array = eval {
            pdl($pdl_type, $source_array);
        };
        confess($@) if $@;
    }
    elsif($source_array->isa('AI::MXNet::NDArray'))
    {
        return $source_array;
    }
    $source_array = pdl($pdl_type, [@{ $source_array->unpdl } ? $source_array->unpdl->[0] : 0 ]) unless @{ $source_array->shape->unpdl };
    return $source_array;
}


# Prepare the value of dtype if `dtype` is undef. If `src_array` is an NDArray, PDL
# or PDL::CCS::Ne, return src_array->dtype. float32 is returned otherwise.

method _prepare_default_dtype($src_array, $dtype)
{
    if(not defined $dtype)
    {
        if(blessed $src_array)
        {
            $dtype = $src_array->dtype;
        }
        else
        {
            $dtype = 'float32';
        }
    }
    return $dtype;
}

use Data::Dumper;
method _check_shape($s1, $s2)
{
    my ($ps1, $ps2) = map { (blessed($_) and $_->isa('AI::MXNet:NDArray')) ? pdl($_->shape) : blessed($_) ? $_ : pdl($_) } ($s1, $s2);
    return 1 unless defined $s2;
    ($ps1 == $ps2)->all
        or
    confess("Shape mismatch detected. " . Dumper(blessed ($s1) ? $s1->undpl : $s1 ) . " v.s. " . Dumper(blessed ($s2) ? $s2 : $s2));
}

method coo_matrix(@args)
{
    my ($data, $row, $col, $shape) = map { blessed $_ ? $_ : pdl($_) } @args;
    my @which;
    my $i = 0;
    my $j = 0;
    for (my $i = 0; $i < $row->nelem; $i++)
    {
        push @which, [$row->at($i), $col->at($i)];
    }
    return PDL::CCS::Nd->newFromWhich(
            pdl(\@which), $data, pdims => $shape
    )->xchg(0, 1);
}

=head2 csr_matrix

    Creates a AI::MXNet::NDArray::CSR, an 2D array with compressed sparse row (CSR) format.

    The AI::MXNet::NDArray::CSR can be instantiated in several ways:

    - csr_matrix($arg1, Maybe[AI::MXNet::Context] :$ctx=, Maybe[Shape] :$shape, Maybe [Dtype] :$dtype=)
        $ctx, $shape, $dtype are optional
        $arg1 can be given in following variants

    - to construct a AI::MXNet::NDArray::CSR with a dense 2D array $arg1
            - $arg1 is in AI::MXNet::NDArray::array input format

    - to construct a AI::MXNet::NDArray::CSR with a sparse 2D array $arg1
            $arg1 is AI::MXNet::NDArray::CSR or PDL::CCS::Nd - A sparse matrix.
            PDL::CCS::Nd is expected to be converted internally into CSR format
            AI::MXNet injects 'tocsr' method into PDL and PDL::CCS::Nd modules for this purpose.

    - to construct an empty AI::MXNet::NDArray::CSR with shape $arg1 = [$M, $N]
            -  $M - Number of rows in the matrix
            -  $N - Number of columns in the matrix

    - to construct a AI::MXNet::NDArray::CSR based on the definition of compressed sparse row format
        using three separate arrays,
        where the column indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
        and their corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
        The column indices for a given row are expected to be **sorted in ascending order.**
        Duplicate column entries for the same row are not allowed.
        In this case $arg1 = [$data, $indices, $indptr]
            $data, $indices, $indptr must be given in the AI::MXNet::NDArray::array input format
            - $data - holds all the non-zero entries of the matrix in row-major order.
            - $indices - stores the column index for each non-zero element in $data.
            stores the column index for each non-zero element in $data.
            - $indptr  - stores the offset into $data of the first non-zero element number of each
            row of the matrix.

        to construct a AI::MXNet::NDArray::CSR based on the COOrdinate format
        using three seperate arrays, 
        where ``row[i]`` is the row index of the element,
        ``col[i]`` is the column index of the element
        and ``data[i]`` is the data corresponding to the element. All the missing
        elements in the input are taken to be zeroes.
        In this case $arg1 = [$data, [$row, $col]]
            $data, $row, $col must be given in the AI::MXNet::NDArray::array input format
            $data - holds all the non-zero entries of the matrix in COO format.
            $row - stores the row index for each non zero element in $data.
            - **col** (*array_like*) - An object exposing the array interface, which
            $col - stores the col index for each non zero element in $data.

    Returns
    -------
    AI::MXNet::NDArray::CSR
        A AI::MXNet::NDArray::CSR with the 'csr' storage representation.

    Example
    -------
    >>> $a = mx->nd->sparse->csr_matrix([[1, 2, 3], [1, 0, 2], [0, 1, 2, 2, 3]], shape => [4, 3])
    >>> $a->aspdl
          [[ 0  1  0]
           [ 2  0  0]
           [ 0  0  0]
           [ 0  0  3]]

    See Also
    --------
    CSRNDArray : MXNet NDArray in compressed sparse row format.
=cut
method csr_matrix(
    $arg1,
    Maybe[Shape|PDL]          :$shape=,
    Maybe[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype]              :$dtype=
)
{
    if(not defined $arg1)
    {
        return __PACKAGE__->empty('csr', $shape, ctx => $ctx, (defined $dtype ? (dtype => $dtype) : ()));
    }
    # construct a csr matrix from (M, N) or (data, indices, indptr)
    if(ref $arg1 eq 'ARRAY')
    {
        my $arg_len = @{ $arg1 };
        if($arg_len == 2)
        {
            # construct a sparse csr matrix from
            # scipy coo matrix if input format is coo
            if(ref $arg1->[1] eq 'ARRAY' and @{ $arg1->[1] } == 2)
            {
                my $coo = __PACKAGE__->coo_matrix($arg1->[0], @{ $arg1->[1] }, $shape);
                __PACKAGE__->_check_shape($coo->shape, $shape);
                return __PACKAGE__->array($coo, ctx => $ctx, dtype => $dtype);
            }
            else
            {
                # empty matrix with shape
                __PACKAGE__->_check_shape($arg1, $shape);
                return __PACKAGE__->empty('csr', $arg1, ctx=>$ctx, dtype=>$dtype);
            }
        }
        elsif($arg_len == 3)
        {
            # data, indices, indptr
            return __PACKAGE__->_csr_matrix_from_definition(
                @{ $arg1 }, shape  => $shape,
                ctx => $ctx, dtype => $dtype
            );
        }
        else
        {
            confess("Unexpected length of input array: " . Dumper($arg1));
        }
    }
    else
    {
        # construct a csr matrix from a sparse / dense one
        if(blessed $arg1 and ($arg1->isa('AI::MXNet::NDArray::CSR') or $arg1->isa('PDL::CCS::Nd')))
        {
            # construct a csr matrix from scipy or CSRNDArray
            __PACKAGE__->_check_shape($arg1->shape, $shape);
            return __PACKAGE__->array($arg1, ctx => $ctx, dtype => $dtype);
        }
        elsif(blessed $arg1 and $arg1->isa('AI::MXNet::NDArray::RowSparse'))
        {
            confess("Unexpected input type: AI::MXNet::NDArray::RowSparse");
        }
        else
        {
            # construct a csr matrix from a dense one
            # prepare default ctx and dtype since mx.nd.array doesn't use default values
            # based on source_array
            $dtype = __PACKAGE__->_prepare_default_dtype($arg1, $dtype);
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            my $dns = __PACKAGE__->array($arg1, dtype=>$dtype);
            if(defined $ctx and $dns->context ne $ctx)
            {
                $dns = $dns->as_in_context($ctx);
            }
            __PACKAGE__->_check_shape($dns->shape, $shape);
            return $dns->tostype('csr');
        }
    }
}

# Create a AI::MXNet::NDarray::CSR based on data, indices and indptr
method _csr_matrix_from_definition(
    $data, $indices, $indptr,
    Maybe[Shape|PDL] :$shape=,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype] :$dtype=,
    Maybe[Dtype] :$indices_type=STORAGE_AUX_TYPES->{'csr'}[0],
    Maybe[Dtype] :$indptr_type=STORAGE_AUX_TYPES->{'csr'}[1]
)
{
    $dtype = __PACKAGE__->_prepare_default_dtype($data, $dtype);
    # prepare src array and types
    $data = __PACKAGE__->_prepare_src_array($data, $dtype);
    $indptr = __PACKAGE__->_prepare_src_array($indptr, $indptr_type);
    $indices = __PACKAGE__->_prepare_src_array($indices, $indices_type);

    if(not (blessed $data and $data->isa('AI::MXNet::NDArray')))
    {
        $data = __PACKAGE__->array($data, ctx => $ctx, dtype => $dtype);
    }
    if(not (blessed $indptr and $indptr->isa('AI::MXNet::NDArray')))
    {
        $indptr = __PACKAGE__->array($indptr, ctx => $ctx, dtype => $indptr_type);
    }
    if(not (blessed $indices and $indices->isa('AI::MXNet::NDArray')))
    {
        $indices = __PACKAGE__->array($indices, ctx => $ctx, dtype => $indices_type);
    }
    if(not defined $shape)
    {
        if($indices->shape->[0] == 0)
        {
            confess('invalid shape');
        }
        $shape = [@{ $indptr } - 1, $indices->max->asscalar + 1];
    }
    elsif(blessed $shape)
    {
        $shape = $shape->unpdl;
    }
    # verify shapes
    my $aux_shapes = [$indptr->shape, $indices->shape];
    if($data->ndim != 1 or $indptr->ndim != 1 or $indices->ndim != 1 or $indptr->shape->[0] == 0 or @{ $shape } != 2)
    {
        confess('invalid shape');
    }
    my $hdl = __PACKAGE__->_new_alloc_handle(
        'csr', $shape, $ctx, 0, $dtype,
        [$indptr_type, $indices_type], $aux_shapes
    );
    my $result = AI::MXNet::NDArray::CSR->new(handle => $hdl);
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromNDArray($result->handle, $data->handle, -1));
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromNDArray($result->handle, $indptr->handle, 0));
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromNDArray($result->handle, $indices->handle, 1));
    return $result;
}

=head2 row_sparse_array

    Creates a AI::MXNet::NDArray::RowSparse, a multidimensional row sparse array with a set of
    tensor slices at given indices.

    The AI::MXNet::NDArray::RowSparse can be instantiated in several ways:

    - row_sparse_array($arg1, Maybe[AI::MXNet::Context] :$ctx=, Maybe[Shape] :$shape, Maybe [Dtype] :$dtype=)
        $ctx, $shape, $dtype are optional
        $arg1 can be given in following variants

    - to construct a AI::MXNet::NDArray::RowSparse with a dense array $arg1
            - $arg1 is in AI::MXNet::NDArray::array input format

    - to construct a AI::MXNet::NDArray::RowSparse with a sparse array $arg1
            $arg1 is AI::MXNet::NDArray::RowSparse

    - to construct an empty AI::MXNet::NDArray::RowSparse with shape $arg1 = [$D1, $D1, ...$DN]

    - to construct a RowSparseNDArray based on the definition of row sparse format
        using two separate arrays,
        where the $indices stores the indices of the row slices with non-zeros,
        while the values are stored in $data. The corresponding NDArray dense
        represented by RowSparse rsp has
        dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]
        The row indices for are expected to be **sorted in ascending order.
        $arg1 = [$data, $indices]
            $data, $indices must be given in the AI::MXNet::NDArray::array input format

    Returns
    -------
    AI::MXNet::NDArray::RowSparse
        A AI::MXNet::NDArray::RowSparse with the 'row_sparse' storage representation.

    Example
    -------
    >>> $a = mx->nd->sparse->row_sparse_array([[[1, 2], [3, 4]], [1, 4]], shape=>[6, 2])
    >>> $a->aspdl
          [[ 0  0]
           [ 1  2]
           [ 0  0]
           [ 0  0]
           [ 3  4]
           [ 0  0]]
=cut

method row_sparse_array(
    $arg1,
    Maybe[Shape]              :$shape=,
    Maybe[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype]              :$dtype=
)
{
    if(not defined $arg1)
    {
        return __PACKAGE__->empty('row_sparse', $shape, ctx => $ctx, (defined $dtype ? (dtype => $dtype) : ()));
    }
    # construct a row sparse array from (D0, D1 ..) or (data, indices)
    if(ref $arg1 eq 'ARRAY')
    {
        my $arg_len = @{ $arg1 };
        if($arg_len < 2)
        {
            confess("Unexpected length of input array: $arg_len ");
        }
        elsif($arg_len > 2)
        {
            # empty ndarray with shape
            __PACKAGE__->_check_shape($arg1, $shape);
            return __PACKAGE__->empty('row_sparse', $arg1, ctx => $ctx, dtype => $dtype);
        }
        else
        {
            # len(arg1) = 2, is either shape or (data, indices)
            if(not ref $arg1->[0] and not ref $arg1->[1])
            {
                # empty ndarray with shape
                __PACKAGE__->_check_shape($arg1, $shape);
                return __PACKAGE__->empty('row_sparse', $arg1, ctx => $ctx, dtype => $dtype);
            }
            else
            {
                # data, indices, indptr
                return __PACKAGE__->_row_sparse_ndarray_from_definition(
                    @{ $arg1 }, shape => $shape, ctx => $ctx, dtype => $dtype
                );
            }
        }
    }
    else
    {
        # construct a row sparse ndarray from a dense / sparse array
        if(blessed $arg1 and $arg1->isa('AI::MXNet::NDArray::RowSparse'))
        {
            # construct a row sparse ndarray from RowSparseNDArray
            __PACKAGE__->_check_shape($arg1->shape, $shape);
            return __PACKAGE__->array($arg1, ctx => $ctx, dtype => $dtype);
        }
        elsif(blessed $arg1 and $arg1->isa('AI::MXNet::NDArray::CSR'))
        {
            confess("Unexpected input type: AI::MXNet::NDArray::CSR");
        }
        else
        {
            # construct a csr matrix from a dense one
            # prepare default dtype since mx.nd.array doesn't use default values
            # based on source_array
            $dtype = __PACKAGE__->_prepare_default_dtype($arg1, $dtype);
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            my $dns = __PACKAGE__->array($arg1, dtype => $dtype);
            if(defined $ctx and $dns->context ne $ctx)
            {
                $dns = $dns->as_in_context($ctx);
            }
            __PACKAGE__->_check_shape($dns->shape, $shape);
            return $dns->tostype('row_sparse');
        }
    }
}

# Create a AI::MXNet::NDArray::RowSparse based on data and indices
method _row_sparse_ndarray_from_definition(
    $data, $indices,
    Maybe[Shape] :$shape=,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype] :$dtype=,
    Maybe[Dtype] :$indices_type=STORAGE_AUX_TYPES->{'row_sparse'}[0]
)
{
    $dtype = __PACKAGE__->_prepare_default_dtype($data, $dtype);
    # prepare src array and types
    $data = __PACKAGE__->_prepare_src_array($data, $dtype);
    $indices = __PACKAGE__->_prepare_src_array($indices, $indices_type);

    if(not (blessed $data and $data->isa('AI::MXNet::NDArray')))
    {
        $data = __PACKAGE__->array($data, ctx => $ctx, dtype => $dtype);
    }
    if(not (blessed $indices and $indices->isa('AI::MXNet::NDArray')))
    {
        $indices = __PACKAGE__->array($indices, ctx => $ctx, dtype => $indices_type);
    }
    if(not defined $shape)
    {
        my $num_indices = $indices->shape->[0];
        if($num_indices == 0)
        {
            confess('invalid shape');
        }
        my $dim0 = $indices->at($num_indices - 1)->asscalar + 1;
        $shape = [$dim0, @{ $data->shape } [1..@{ $data->shape } - 1]];
    }
    # verify shapes
    if($data->ndim != @{ $shape } or $indices->ndim != 1 or product(@{ $shape } [1..@{ $shape } - 1]) == 0)
    {
        confess("invalid shape");
    }
    my $handle = __PACKAGE__->_new_alloc_handle(
        'row_sparse', $shape, $ctx, 0, $dtype,
        [$indices_type], [$indices->shape]
    );
    my $result = AI::MXNet::NDArray::RowSparse->new(handle => $handle);
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromNDArray($result->handle, $data->handle, -1));
    check_call(AI::MXNetCAPI::NDArraySyncCopyFromNDArray($result->handle, $indices->handle, 0));
    return $result
}

=head2 zeros

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    $stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or array ref of int
        The shape of the empty array
    :$ctx : AI::MXNet::Context, optional
        An optional device context (default is the current default context)
    :$dtype : Dtype, optional
        An optional value type (default is `float32`)

    Returns
    -------
    AI::MXNet::NDArray::RowSparse or AI::MXNet::NDArray::CSR
        A created array
    Examples
    --------
    >>> mx->nd->sparse->zeros('csr', [1,2])
    <AI::MXNet::NDArray::CSR 1x2 @cpu(0)>
    >>> mx->nd->sparse->zeros('row_sparse', [1,2], ctx=>mx->cpu(), dtype=>'float16')->aspdl
    [[ 0  0]]
=cut


method zeros(
    Stype $stype,
    Shape $shape,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype] :$dtype='float32',
    Maybe[AI::MXNet::NDArray] :$out=,
    Maybe[Str] :$name=,
    Maybe[Str] :$__layout__=
)
{
    if($stype eq 'default')
    {
        return AI::MXNet::NDArray->zeros(
            $shape, ctx => $ctx, dtype => $dtype, out => $out, name => $name, __layout__ => $__layout__
        );
    }
    my $aux_types;
    if($stype eq 'row_sparse' or $stype or 'csr')
    {
        $aux_types = STORAGE_AUX_TYPES->{ $stype };
    }
    else
    {
        confess("unknown storage type: $stype");
    }
    $out //= __PACKAGE__->_ndarray_cls(
        __PACKAGE__->_new_alloc_handle(
            $stype, $shape, $ctx, 1, $dtype, $aux_types)
    );
    return __PACKAGE__->_zeros(
        shape => $shape, ctx => $ctx, dtype => $dtype, out => $out,
        ($__layout__ ? (__layout__ => $__layout__) : ())
    );
}

=head2 empty

    Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or array ref of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : Dtype, optional
        An optional value type (default is `float32`).

    Returns
    -------
    AI::MXNet::NDArray::CSR or AI::MXNet::NDArray::RowSparse
        A created array.
=cut

method empty(
    Stype $stype,
    Shape $shape,
    Maybe[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype] :$dtype='float32'
)
{
    assert(defined $stype);
    return __PACKAGE__->zeros($stype, $shape, ctx => $ctx, dtype => $dtype);
}

=head2 array

    Creates a sparse array from any object exposing the array interface.

    Parameters
    ----------
    $source_array : AI::MXNet::NDArray::RowSparse, AI::MXNet::NDArray::CSR or PDL::CCS::Nd
        The source sparse array
    :$ctx : Context, optional
        The default context is $source_array->context if $source_array is an NDArray.
        The current default context otherwise.
    :$dtype : Dtype, optional
        The data type of the output array. The default dtype is $source_array->dtype
        if $source_array is an AI::MXNet::NDArray, PDL or PDL::CCS::Nd
        'float32' otherwise.

    Returns
    -------
    AI::MXNet::NDArray::RowSparse or AI::MXNet::NDArray::CSR
        An array with the same contents as the $source_array.

    Examples
    --------
    >>> use PDL; use PDL::CCS::Nd
    >>> $csr = zeros([100, 2])->tocsr
    >>> mx->nd->sparse->array($csr)
    <AI::MXNet::NDArray::CSR 2x100 @cpu(0)>
    >>> mx->nd->sparse->array(mx->nd->sparse->zeros('csr', [3, 2]))
    <AI::MXNet::NDArray::CSR 3x2 @cpu(0)>
    >>> mx->nd->sparse->array(mx->nd->sparse->zeros('row_sparse', [3, 2]))
    <AI::MXNet::NDArray::RowSparse 3x2 @cpu(0)>
=cut

method array(
    AcceptableInput $source_array,
    Maybe[AI::MXNet::Context] :$ctx=AI::MXNet::Context->current_ctx,
    Maybe[Dtype] :$dtype='float32',
    Maybe[PDL] :$pdl=
)
{
    if(not blessed $source_array  or $source_array->isa('PDL') or ($source_array->isa('AI::MXNet::NDArray') and $source_array->stype eq 'default'))
    {
        if(not ref $source_array)
        {
            $pdl .= $source_array;
            $source_array = $pdl;
        }
        return __PACKAGE__->SUPER::array($source_array, ctx => $ctx, dtype => $dtype);
    }
    if($source_array->isa('AI::MXNet::NDArray'))
    {
        assert(
            ($source_array->stype ne 'default'),
            "Please use tostype to create AI::MXNet::NDarray::RowSparse or AI::MXNet::NDarray::CSR from an AI::MXNet::NDarray"
        );
        # prepare dtype and ctx based on source_array, if not provided
        $dtype = __PACKAGE__->_prepare_default_dtype($source_array, $dtype);
        # if both dtype and ctx are different from source_array, we cannot copy directly
        my $arr;
        if($source_array->dtype ne $dtype and $source_array->context ne $ctx)
        {
            $arr = __PACKAGE__->empty($source_array->stype, $source_array->shape, dtype => $dtype);
            $arr .= $source_array;
            $arr = $arr->as_in_context($ctx);
        }
        else
        {
            $arr = __PACKAGE__->empty($source_array->stype, $source_array->shape, dtype => $dtype, ctx => $ctx);
            $arr .= $source_array;
        }
        return $arr;
    }
    elsif($source_array->isa('PDL::CCS::Nd'))
    {
        $dtype = __PACKAGE__->_prepare_default_dtype($source_array, $dtype);
        return __PACKAGE__->csr_matrix(
            [$source_array->data, $source_array->indices, $source_array->indptr],
            shape => $source_array->shape , dtype => $dtype, ctx => $ctx
        );
    }
}

sub AUTOLOAD {
    my $sub = $AI::MXNet::NDArray::Sparse::AUTOLOAD;
    $sub =~ s/.*:://;
    $sub = "_sparse_$sub";
    shift;
    return AI::MXNet::NDArray->$sub(@_);
}

1;
