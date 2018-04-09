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

package AI::MXNet::TestUtils;
use strict;
use warnings;
use PDL;
use Carp qw(confess);
use Scalar::Util qw(blessed);
use List::Util qw(shuffle);
use AI::MXNet::Function::Parameters;
use AI::MXNet::Base;
use Exporter;
use base qw(Exporter);
@AI::MXNet::TestUtils::EXPORT_OK = qw(same reldiff almost_equal GetMNIST_ubyte
                                      GetCifar10 pdl_maximum pdl_minimum mlp2 conv dies_ok
                                      check_consistency zip assert enumerate same_array dies_like allclose rand_shape_2d
                                      rand_shape_3d rand_sparse_ndarray random_arrays rand_ndarray randint pdl);
use constant default_numerical_threshold => 1e-6;
=head1 NAME

    AI::MXNet::TestUtils - Convenience subs used in tests.

=head2 same

    Test if two pdl arrays are the same

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func same(PDL $a, PDL $b)
{
    return ($a != $b)->sum == 0;
}

=head2 allclose

    Test if all elements of two pdl arrays are almost equal

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func allclose(PDL $a, PDL $b, Maybe[Num] $threshold=)
{
    return (($a - $b)->abs <= ($threshold//default_numerical_threshold))->all;
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

func _pdl_compare(PDL $a, PDL|Num $b, Str $criteria)
{
    if(not blessed $b)
    {
        my $tmp = $b;
        $b = $a->copy;
        $b .= $tmp;
    }
    my $mask = {
        'max' => sub { $_[0] < $_[1] },
        'min' => sub { $_[0] > $_[1] },
    }->{$criteria}->($a, $b);
    my $c = $a->copy;
    $c->where($mask) .= $b->where($mask);
    $c;
}

func pdl_maximum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'max');
}

func pdl_minimum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'min');
}

func mlp2()
{
    my $data = AI::MXNet::Symbol->Variable('data');
    my $out  = AI::MXNet::Symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>1000);
    $out     = AI::MXNet::Symbol->Activation(data=>$out, act_type=>'relu');
    $out     = AI::MXNet::Symbol->FullyConnected(data=>$out, name=>'fc2', num_hidden=>10);
    return $out;
}

func conv()
{
    my $data    = AI::MXNet::Symbol->Variable('data');
    my $conv1   = AI::MXNet::Symbol->Convolution(data => $data, name=>'conv1', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn1     = AI::MXNet::Symbol->BatchNorm(data => $conv1, name=>"bn1");
    my $act1    = AI::MXNet::Symbol->Activation(data => $bn1, name=>'relu1', act_type=>"relu");
    my $mp1     = AI::MXNet::Symbol->Pooling(data => $act1, name => 'mp1', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $conv2   = AI::MXNet::Symbol->Convolution(data => $mp1, name=>'conv2', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn2     = AI::MXNet::Symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2    = AI::MXNet::Symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2     = AI::MXNet::Symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $fl      = AI::MXNet::Symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc2     = AI::MXNet::Symbol->FullyConnected(data => $fl, name=>'fc2', num_hidden=>10);
    my $softmax = AI::MXNet::Symbol->SoftmaxOutput(data => $fc2, name => 'sm');
    return $softmax;
}

=head2 check_consistency

    Check symbol gives the same output for different running context

    Parameters
    ----------
    sym : Symbol or list of Symbols
        symbol(s) to run the consistency test
    ctx_list : list
        running context. See example for more detail.
    scale : float, optional
        standard deviation of the inner normal distribution. Used in initialization
    grad_req : str or list of str or dict of str to str
        gradient requirement.
=cut

my %dtypes = (
    float32 => 0,
    float64 => 1,
    float16 => 2,
    uint8   => 3,
    int32   => 4
);

func check_consistency(
    SymbolOrArrayOfSymbols              :$sym,
    ArrayRef                            :$ctx_list,
    Num                                 :$scale=1,
    Str|ArrayRef[Str]|HashRef[Str]      :$grad_req='write',
    Maybe[HashRef[AI::MXNet::NDArray]]  :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]]  :$aux_params=,
    Maybe[HashRef[Num]|Num]             :$tol=,
    Bool                                :$raise_on_err=1,
    Maybe[AI::MXNer::NDArray]           :$ground_truth=
)
{
    $tol //= {
        float16 => 1e-1,
        float32 => 1e-3,
        float64 => 1e-5,
        uint8   => 0,
        int32   => 0
    };
    $tol = {
        float16 => $tol,
        float32 => $tol,
        float64 => $tol,
        uint8   => $tol,
        int32   => $tol
    } unless ref $tol;

    Test::More::ok(@$ctx_list > 1);
    if(blessed $sym)
    {
        $sym = [($sym)x@$ctx_list];
    }
    else
    {
        Test::More::ok(@$sym == @$ctx_list);
    }
    my $output_names = $sym->[0]->list_outputs;
    my $arg_names    = $sym->[0]->list_arguments;
    my @exe_list;
    zip(sub {
        my ($s, $ctx) = @_;
        Test::More::is_deeply($s->list_arguments, $arg_names);
        Test::More::is_deeply($s->list_outputs, $output_names);
        push @exe_list, $s->simple_bind(grad_req=>$grad_req, %$ctx);
    }, $sym, $ctx_list);
    $arg_params //= {};
    $aux_params //= {};
    my %arg_dict = %{ $exe_list[0]->arg_dict };
    while(my ($n, $arr) = each %arg_dict)
    {
        if(not exists $arg_params->{$n})
        {
            $arg_params->{$n} = random(reverse @{ $arr->shape })*$scale;
        }
    }
    my %aux_dict = %{ $exe_list[0]->aux_dict };
    while(my ($n, $arr) = each %aux_dict)
    {
        if(not exists $aux_params->{$n})
        {
            $aux_params->{$n} = 0;
        }
    }
    for my $exe(@exe_list)
    {
        %arg_dict = %{ $exe->arg_dict };
        while(my ($name, $arr) = each %arg_dict)
        {
            $arr .= $arg_params->{$name};
        }
        %aux_dict = %{ $exe->aux_dict };
        while(my ($name, $arr) = each %aux_dict)
        {
            $arr .= $aux_params->{$name};
        }
    }
    my @dtypes = map { $_->outputs->[0]->dtype } @exe_list;
    my $max_idx = pdl(map { $dtypes{$_} } @dtypes)->maximum_ind;
    my $gt = $ground_truth;
    if(not defined $gt)
    {
        $gt = { %{ $exe_list[$max_idx]->output_dict } };
        if($grad_req ne 'null')
        {
            %{$gt} = (%{$gt}, %{ $exe_list[$max_idx]->grad_dict });
        }
    }

    # test
    for my $exe (@exe_list)
    {
        $exe->forward(0);
    }
    enumerate(sub {
        my ($i, $exe) = @_;
        if($i == $max_idx)
        {
            return;
        }
        zip(sub {
            my ($name, $arr) = @_;
            my $gtarr = $gt->{$name}->astype($dtypes[$i])->aspdl;
            $arr = $arr->aspdl;
            Test::More::ok(
                almost_equal(
                    $arr, $gtarr,
                    $tol->{$dtypes[$i]}
                )
            );
        }, $output_names, $exe->outputs);
    }, \@exe_list);

    # train
    if ($grad_req ne 'null')
    {
        for my $exe (@exe_list)
        {
            $exe->forward(1);
            $exe->backward($exe->outputs);
        }
        enumerate(sub {
            my ($i, $exe) = @_;
            return if($i == $max_idx);
            zip(sub {
                my ($name, $arr) = @_;
                if (not defined $gt->{$name})
                {
                    Test::More::ok(not defined $arr);
                    return;
                }
                my $gtarr = $gt->{$name}->astype($dtypes[$i])->aspdl;
                $arr = $arr->aspdl;
                Test::More::ok(
                    almost_equal(
                        $arr, $gtarr,
                        $tol->{$dtypes[$i]}
                    )
                );
            }, [@$output_names, @$arg_names], [@{ $exe->outputs }, @{ $exe->grad_arrays }]);
        }, \@exe_list);
    }
    return $gt;
}

=head2 same_array

    Check whether two NDArrays sharing the same memory block

    Parameters
    ----------

    array1 : NDArray
        First NDArray to be checked
    array2 : NDArray
        Second NDArray to be checked

    Returns
    -------
    bool
        Whether two NDArrays share the same memory
=cut

func same_array(
    AI::MXNet::NDArray $array1,
    AI::MXNet::NDArray $array2
)
{
    $array1 += 1;
    if(not same($array1->aspdl, $array2->aspdl))
    {
        $array1 -= 1;
        return 0
    }
    $array1 -= 1;
    return same($array1->aspdl, $array2->aspdl);
}

func dies_like($code, $regexp)
{
    eval { $code->() };
    if($@ =~ $regexp)
    {
        return 1;
    }
    else
    {
        warn $@;
        return 0;
    }
}

func random_arrays(@shapes)
{
    my @arrays = map { random(reverse(@$_))->float } @shapes;
    if(@arrays > 1)
    {
        return @arrays;
    }
    else
    {
        return $arrays[0];
    }
}


func _validate_csr_generation_inputs(
    $num_rows, $num_cols, $density,
    $distribution="uniform"
)
{
    my $total_nnz = int($num_rows * $num_cols * $density);
    if($density < 0 or $density > 1)
    {
        confess("density has to be between 0 and 1");
    }
    if($num_rows <= 0 or $num_cols <= 0)
    {
        confess("num_rows or num_cols should be greater than 0");
    }
    if($distribution eq "powerlaw")
    {
        if($total_nnz < 2 * $num_rows)
        {
            confess(
                "not supported for this density: $density"
                ." for this shape ($num_rows, $num_cols)"
                ." Please keep :"
                ." num_rows * num_cols * density >= 2 * num_rows"
            );
        }
    }
}

# Shuffle CSR column indices per row
# This allows validation of unordered column indices, which is not a requirement
# for a valid CSR matrix

func shuffle_csr_column_indices($csr)
{
    my $row_count = @{ $csr->indptr } - 1;
    for my $i (0..$row_count-1)
    {
        my $start_index = $csr->indptr->[$i];
        my $end_index   = $csr->indptr->[$i + 1];
        my @sublist = @{$csr->indices}[$start_index .. $end_index];
        @sublist = shuffle(@sublist);
        @{$csr->indices}[$start_index .. $end_index] = @sublist;
    }
}


func _get_uniform_dataset_csr(
    $num_rows, $num_cols, $density=0.1, $dtype='float32',
    $data_init=, $shuffle_csr_indices=0)
{
    # Returns CSRNDArray with uniform distribution
    # This generates a csr matrix with totalnnz unique randomly chosen numbers
    # from num_rows*num_cols and arranges them in the 2d array in the
    # following way:
    # row_index = (random_number_generated / num_rows)
    # col_index = random_number_generated - row_index * num_cols

    _validate_csr_generation_inputs(
        $num_rows, $num_cols, $density,
        "uniform"
    );
    my $csr = rand_sparse($num_rows, $num_cols, $density, $dtype, "csr");
    if(defined $data_init)
    {
        $csr->data->fill($data_init);
    }
    if($shuffle_csr_indices)
    {
        shuffle_csr_column_indices($csr);
    }
    return mx->nd->sparse->csr_matrix(
        [$csr->data, $csr->indices, $csr->indptr],
        shape => [$num_rows, $num_cols], dtype => $dtype
    );
}

func _get_powerlaw_dataset_csr($num_rows, $num_cols, $density=0.1, $dtype='float32')
{
    # Returns CSRNDArray with powerlaw distribution
    # with exponentially increasing number of non zeros in each row.
    # Not supported for cases where total_nnz < 2*num_rows. This is because
    # the algorithm first tries to ensure that there are rows with no zeros by
    # putting non zeros at beginning of each row.

    _validate_csr_generation_inputs($num_rows, $num_cols, $density,
                                    "powerlaw");

    my $total_nnz = int($num_rows * $num_cols * $density);

    my $unused_nnz = $total_nnz;
    my $output_arr = zeros($num_cols, $num_rows);
    # Start with ones on each row so that no row is empty
    for my $row (0..$num_rows-1)
    {
        $output_arr->slice(0, $row) .= 1 + rand(2);
        $unused_nnz--;
        if($unused_nnz <= 0)
        {
            return mx->nd->array($output_arr)->tostype("csr");
        }
    }
    # Populate rest of matrix with 2^i items in ith row.
    # if we have used all total nnz return the sparse matrix
    # else if we reached max column size then fill up full columns until we use all nnz
    my $col_max = 2;
    for my $row (0..$num_rows-1)
    {
        my $col_limit = List::Util::min($num_cols, $col_max);
        # In case col_limit reached assign same value to all elements, which is much faster
        if($col_limit == $num_cols and $unused_nnz > $col_limit)
        {
            $output_arr->slice('X', $row) .= 1 + rand(2);
            $unused_nnz = $unused_nnz - $col_limit + 1;
            if($unused_nnz <= 0)
            {
                return mx->nd->array($output_arr)->tostype("csr");
            }
        }
        else
        {
            for my $col_index (1..$col_limit-1)
            {
                $output_arr->slice($col_index, $row) .= 1 + rand(2);
                $unused_nnz--;
                if($unused_nnz <= 0)
                {
                    return mx->nd->array($output_arr)->tostype("csr");
                }
            }
            $col_max *= 2;
        }
    }

    if($unused_nnz > 0)
    {
        warn $unused_nnz;
        confess(
            "not supported for this density: $density"
            ." for this shape ($num_rows,$num_cols)"
        );
    }
    else
    {
        return mx->nd->array($output_arr)->tostype("csr");
    }
}


func assign_each($input, $function=)
{
    my $res = pdl($input);
    if(defined $function)
    {
        return $function->($res);
    }
    return $res;
}

func assign_each2($input1, $input2, $function=)
{
    my $res = pdl($input1);
    if(defined $function)
    {
        return $function->($res, pdl($input2));
    }
    return $res;
}

=head2 rand_sparse_ndarray

    Generate a random sparse ndarray. Returns the ndarray, value(np) and indices(np)

    Parameters
    ----------
    shape: list or tuple
    stype: str, valid values: "csr" or "row_sparse"
    density, optional: float, should be between 0 and 1
    distribution, optional: str, valid values: "uniform" or "powerlaw"
    dtype, optional: numpy.dtype, default value is None

    Returns
    -------
    Result of type CSRNDArray or RowSparseNDArray

    Examples
    --------
    Below is an example of the powerlaw distribution with csr as the stype.
    It calculates the nnz using the shape and density.
    It fills up the ndarray with exponentially increasing number of elements.
    If there are enough unused_nnzs, n+1th row will have twice more nnzs compared to nth row.
    else, remaining unused_nnzs will be used in n+1th row
    If number of cols is too small and we have already reached column size it will fill up
    all following columns in all followings rows until we reach the required density.

    >>> csr_arr, _ = rand_sparse_ndarray(shape=(5, 16), stype="csr",
                                         density=0.50, distribution="powerlaw")
    >>> indptr = csr_arr.indptr.asnumpy()
    >>> indices = csr_arr.indices.asnumpy()
    >>> data = csr_arr.data.asnumpy()
    >>> row2nnz = len(data[indptr[1]:indptr[2]])
    >>> row3nnz = len(data[indptr[2]:indptr[3]])
    >>> assert(row3nnz == 2*row2nnz)
    >>> row4nnz = len(data[indptr[3]:indptr[4]])
    >>> assert(row4nnz == 2*row3nnz)
=cut

func rand_sparse_ndarray(
    $shape, $stype, :$density=rand, :$dtype='float32', :$distribution='uniform',
    :$data_init=, :$rsp_indices=, :$modifier_func=,
    :$shuffle_csr_indices=0
)
{
    if($stype eq 'row_sparse')
    {
        assert (
            ($distribution eq "uniform"),
            "Distribution $distribution not supported for row_sparse"
        );
        # sample index
        my $indices;
        if(defined $rsp_indices)
        {
            $indices = $rsp_indices;
            assert($indices->nelem <= $shape->[0]);
        }
        else
        {
            my $idx_sample = random($shape->[0]);
            $indices = which($idx_sample < $density);
        }
        if($indices->shape(-1)->at(0) == 0)
        {
            my $result = mx->nd->zeros($shape, stype=>'row_sparse', dtype=>$dtype);
            return ($result, [pdl([]), pdl([])]);
        }
        # generate random values
        my $val = random(PDL::Type->new(DTYPE_MX_TO_PDL->{ $dtype }), reverse($indices->shape(-1)->at(0), @{ $shape }[1..@{ $shape }-1]));

        # Allow caller to override or adjust random values
        if(defined $data_init)
        {
            $val .= $data_init;
        }
        if(defined $modifier_func)
        {
            $val = assign_each($val, $modifier_func);
        }
        my $arr = mx->nd->sparse->row_sparse_array([$val, $indices], shape=>$shape, dtype=>$dtype);
        return ($arr, [$val, $indices]);
    }
    elsif($stype eq 'csr')
    {
        assert(@{ $shape } == 2);
        my $csr;
        if($distribution eq "uniform")
        {
            $csr = _get_uniform_dataset_csr(
                @{ $shape }, $density, $dtype,
                $data_init, $shuffle_csr_indices
            );
            return ($csr, [$csr->indptr, $csr->indices, $csr->data]);
        }
        elsif($distribution eq "powerlaw")
        {
            $csr = _get_powerlaw_dataset_csr(@{ $shape }, $density, $dtype);
            return ($csr, [$csr->indptr, $csr->indices, $csr->data]);
        }
        else
        {
            confess("Distribution not supported: $distribution");
        }
    }
    else
    {
        confess("unknown storage type");
    }
}

func rand_ndarray(
    $shape, $stype, $density=rand, $dtype='float32',
    $modifier_func=, $shuffle_csr_indices=0, $distribution='uniform'
)
{
    my $arr;
    if($stype eq 'default')
    {
        $arr = mx->nd->array(random_arrays($shape), dtype=>$dtype);
    }
    else
    {
        ($arr) = rand_sparse_ndarray(
            $shape, $stype, density => $density, dtype => $dtype,
            modifier_func => $modifier_func,
            shuffle_csr_indices => $shuffle_csr_indices, distribution => $distribution
        );
    }
    return $arr;
}


func create_sparse_array(
    $shape, $stype, $data_init=, $rsp_indices=,
    $dtype=, $modifier_func=, $density=0.5,
    $shuffle_csr_indices=0
)
{
    my $arr_data;
    if($stype eq 'row_sparse')
    {
        my $arr_indices;
        if(defined $rsp_indices)
        {
            $arr_indices = pdl($rsp_indices);
            $arr_indices->inplace->qsort;
        }
        ($arr_data) = rand_sparse_ndarray(
            $shape, $stype,
            $density, $dtype,
            $data_init,
            $arr_indices,
            $modifier_func
        );
    }
    elsif($stype eq 'csr')
    {
        ($arr_data) = rand_sparse_ndarray(
            $shape,
            $stype,
            $density, $dtype,
            $data_init,
            $modifier_func,
            $shuffle_csr_indices
        );
    }
    else
    {
        confess("Unknown storage type: $stype");
    }
    return $arr_data;
}


func create_sparse_array_zd(
    $shape, $stype, $density, $data_init=,
    $rsp_indices=, $dtype=, $modifier_func=,
    $shuffle_csr_indices=0
)
{
    if($stype eq 'row_sparse')
    {
        $density = 0;
        if(defined $rsp_indices)
        {
            assert($rsp_indices->len <= $shape->[0]);
        }
    }
    return create_sparse_array(
        $shape, $stype,
        $data_init,
        $rsp_indices,
        $dtype,
        $modifier_func,
        $density,
        $shuffle_csr_indices
    );
}

func rand_shape_2d($dim0=10, $dim1=10)
{
    [int(rand($dim0)+1), int(rand($dim1)+1)];
}


func rand_shape_3d($dim0=10, $dim1=10, $dim2=10)
{
    [int(rand($dim0)+1), int(rand($dim1)+1), int(rand($dim1)+1)];
}


func rand_shape_nd($num_dim, $dim=10)
{
    (random($num_dim)*$dim+1)->floor->unpdl;
}

func randint($low=0, $high=10)
{
    my $value = int(rand($high));
    return $value < $low ? $low : $value;
}

sub dies_ok
{
    my $sub = shift;
    eval { $sub->() };
    if($@)
    {
        Test::More::ok(1);
    }
    else
    {
        Test::More::ok(0);
    }
}

1;
