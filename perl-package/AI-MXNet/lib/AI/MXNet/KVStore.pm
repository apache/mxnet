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

package AI::MXNet::KVStore;
use strict;
use warnings;
use AI::MXNet::NS;
use AI::MXNet::Base;
use AI::MXNet::NDArray;
use AI::MXNet::Optimizer;
use MIME::Base64;
use Storable;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::KVStore - Key value store interface of MXNet.

=head1 DESCRIPTION

    Key value store interface of MXNet for parameter synchronization, over multiple devices.
=cut

has 'handle' => (is => 'ro', isa => 'KVStoreHandle', required => 1);
has '_updater' => (is => 'rw',  isa => 'AI::MXNet::Updater');

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::KVStoreFree(shift->handle));
}

=head2  init

    Initialize a single or a sequence of key-value pairs into the store.
    For each key, one must init it before push and pull.
    Only worker 0's (rank == 0) data are used.
    This function returns after data have been initialized successfully

    Parameters
    ----------
    $key : Str|ArrayRef[Str]
        The keys.
    $value : AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]]
        The values.

    Examples
    --------
    >>> # init a single key-value pair
    >>> $shape = [2,3]
    >>> $kv = mx->kv->create('local')
    >>> $kv->init(3, mx->nd->ones($shape)*2)
    >>> $a = mx->nd->zeros($shape)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
    [[ 2  2  2]
    [ 2  2  2]]

    >>> # init a list of key-value pairs
    >>> $keys = [5, 7, 9]
    >>> $kv->init(keys, [map { mx->nd->ones($shape) } 0..@$keys-1])
=cut

method init(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $value
)
{
    my ($keys, $vals) = _key_value($key, $value);
    check_call(
        AI::MXNetCAPI::KVStoreInitEx(
            $self->handle, scalar(@{ $keys }), $keys, $vals
        )
    );
}

=head2  push

    Push a single or a sequence of key-value pairs into the store.
    Data consistency:
    1. this function returns after adding an operator to the engine.
    2. push is always called after all previous push and pull on the same
        key are finished.
    3. there is no synchronization between workers. One can use _barrier()
    to sync all workers.

    Parameters
    ----------
    $key : Str|ArrayRef[Str]
    $value : AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]]
    :$priority=0 : Int, optional
        The priority of the push operation.
        The higher the priority, the faster this action is likely
        to be executed before other push actions.

    Examples
    --------
    >>> # push a single key-value pair
    >>> $kv->push(3, mx->nd->ones($shape)*8)
    >>> $kv->pull(3, out=>$a) # pull out the value
    >>> print $a->aspdl()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

    >>> # aggregate the value and the push
    >>> $gpus = [map { mx->gpu($_) } 0..3]
    >>> $b = [map { mx->nd->ones($shape, ctx => $_) } @$gpus]
    >>> $kv->push(3, $b)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

    >>> # push a list of keys.
    >>> # single device
    >>> $kv->push($keys, [map { mx->nd->ones($shape) } 0..@$keys-1)
    >>> $b = [map { mx->nd->zeros(shape) } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1]->aspdl
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

    >>> # multiple devices:
    >>> $b = [map { [map { mx->nd->ones($shape, ctx => $_) } @$gpus] } @$keys-1]
    >>> $kv->push($keys, $b)
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1][1]->aspdl()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
=cut

method push(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $value,
    Int :$priority=0
)
{
    my ($keys, $vals) = _key_value($key, $value);
    check_call(
        AI::MXNetCAPI::KVStorePushEx(
            $self->handle, scalar(@{ $keys }), $keys, $vals, $priority
        )
    );
}

=head2 pull

    Pull a single value or a sequence of values from the store.

    Data consistency:

    1. this function returns after adding an operator to the engine. But any
        further read on out will be blocked until it is finished.
    2. pull is always called after all previous push and pull on the same
        key are finished.
    3. It pulls the newest value from the store.

    Parameters
    ----------
    $key : Str|ArrayRef[Str]
        Keys
    :$out: AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]]
        According values

    :$priority=0 : Int, optional
        The priority of the push operation.
        The higher the priority, the faster this action is likely
        to be executed before other push actions.

    Examples
    --------
    >>> # pull a single key-value pair
    >>> $a = mx->nd->zeros($shape)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

    >>> # pull into multiple devices
    >>> $b = [map { mx->nd->ones($shape, $_) } @$gpus]
    >>> $kv->pull(3, out=>$b)
    >>> print $b->[1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

    >>> # pull a list of key-value pairs.
    >>> # On single device
    >>> $keys = [5, 7, 9]
    >>> $b = [map { mx->nd->zeros($shape) } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
    >>> # On multiple devices
    >>> $b = [map { [map { mx->nd->ones($shape, ctx => $_) } @$gpus ] } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1][1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
=cut

method pull(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] :$out,
    Int :$priority=0,
    Bool :$ignore_sparse=1
)
{
    my ($keys, $vals) = _key_value($key, $out);
    check_call(
        AI::MXNetCAPI::KVStorePullWithSparseEx(
            $self->handle, scalar(@{ $keys }), $keys, $vals, $priority, $ignore_sparse
        )
    );
}

=head2  row_sparse_pull

        Pulls a single AI::MXNet::NDArray::RowSparse value or an array ref of AI::MXNet::NDArray::RowSparse values
        from the store with specified row_ids. When there is only one row_id, KVStoreRowSparsePull
        is invoked just once and the result is broadcast to all the rest of outputs.

        `row_sparse_pull` is executed asynchronously after all previous
        `pull`/`row_sparse_pull` calls and the last `push` call for the
        same input key(s) are finished.

        The returned values are guaranteed to be the latest values in the store.

        Parameters
        ----------
        $key : Str|ArrayRef[Str] $key
            Keys.

        :$out: AI::MXNet::NDArray::RowSparse|ArrayRef[AI::MXNet::NDArray::RowSparse]|ArrayRef[ArrayRef[AI::MXNet::NDArray::RowSparse]]
            Values corresponding to the keys. The stype is expected to be row_sparse

        :$priority=0 : Int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions.

        :$row_ids : AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]]
            The row_ids for which to pull for each value. Each row_id is an 1D NDArray
            whose values don't have to be unique nor sorted.

        Examples
        --------
        >>> $shape = [3, 3]
        >>> $kv->init('3', mx->nd->ones($shape)->tostype('row_sparse'))
        >>> $a = mx->nd->sparse->zeros('row_sparse', $shape)
        >>> $row_ids = mx->nd->array([0, 2], dtype=>'int64')
        >>> $kv->row_sparse_pull('3', out=>$a, row_ids=>$row_ids)
        >>> print $a->aspdl
        [[ 1.  1.  1.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> $duplicate_row_ids = mx->nd->array([2, 2], dtype=>'int64')
        >>> $kv->row_sparse_pull('3', out=>$a, row_ids=>$duplicate_row_ids)
        >>> print $a->aspdl
        [[ 0.  0.  0.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> $unsorted_row_ids = mx->nd->array([1, 0], dtype=>'int64')
        >>> $kv->row_sparse_pull('3', out=>$a, row_ids=>$unsorted_row_ids)
        >>> print $a->aspdl
        [[ 1.  1.  1.]
        [ 1.  1.  1.]
        [ 0.  0.  0.]]
=cut


method row_sparse_pull(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray::RowSparse|ArrayRef[AI::MXNet::NDArray::RowSparse]|ArrayRef[ArrayRef[AI::MXNet::NDArray::RowSparse]] :$out,
    Int :$priority=0,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] :$row_ids
)
{
    if(blessed $row_ids)
    {
        $row_ids = [$row_ids];
    }
    my $first_out = $out;
    # whether row_ids are the same
    my $single_rowid = 0;
    if(@$row_ids == 1 and ref $out eq 'ARRAY')
    {
        $single_rowid = 1;
        $first_out = [$out->[0]];
    }
    my ($ckeys, $cvals) = _key_value($key, $first_out);
    my (undef, $crow_ids) = _key_value($key, $row_ids);
    assert(
        (@$crow_ids == @$cvals),
        "the number of row_ids doesn't match the number of values"
    );
    check_call(
        AI::MXNetCAPI::KVStorePullRowSparseEx(
            $self->handle, scalar(@$ckeys), $ckeys, $cvals, $crow_ids, $priority
        )
    );
    # the result can be copied to other devices without invoking row_sparse_pull
    # if the indices are the same
    if($single_rowid)
    {
        for my $out_i (@{ $out } [1..@{ $out }-1])
        {
            $out->[0]->copyto($out_i);
        }
    }
}

=head2  set_gradient_compression

        Specifies type of low-bit quantization for gradient compression \
         and additional arguments depending on the type of compression being used.

        2bit Gradient Compression takes a positive float `threshold`.
        The technique works by thresholding values such that positive values in the
        gradient above threshold will be set to threshold. Negative values whose absolute
        values are higher than threshold, will be set to the negative of threshold.
        Values whose absolute values are less than threshold will be set to 0.
        By doing so, each value in the gradient is in one of three states. 2bits are
        used to represent these states, and every 16 float values in the original
        gradient can be represented using one float. This compressed representation
        can reduce communication costs. The difference between these thresholded values and
        original values is stored at the sender's end as residual and added to the
        gradient in the next iteration.

        When kvstore is 'local', gradient compression is used to reduce communication
        between multiple devices (gpus). Gradient is quantized on each GPU which
        computed the gradients, then sent to the GPU which merges the gradients. This
        receiving GPU dequantizes the gradients and merges them. Note that this
        increases memory usage on each GPU because of the residual array stored.

        When kvstore is 'dist', gradient compression is used to reduce communication
        from worker to sender. Gradient is quantized on each worker which
        computed the gradients, then sent to the server which dequantizes
        this data and merges the gradients from each worker. Note that this
        increases CPU memory usage on each worker because of the residual array stored.
        Only worker to server communication is compressed in this setting.
        If each machine has multiple GPUs, currently this GPU to GPU or GPU to CPU communication
        is not compressed. Server to worker communication (in the case of pull)
        is also not compressed.

        To use 2bit compression, we need to specify `type` as `2bit`.
        Only specifying `type` would use default value for the threshold.
        To completely specify the arguments for 2bit compression, we would need to pass
        a dictionary which includes `threshold` like:
        {'type': '2bit', 'threshold': 0.5}

        Parameters
        ----------
        $compression_params : HashRef[Str]
            A dictionary specifying the type and parameters for gradient compression.
            The key `type` in this dictionary is a
            required string argument and specifies the type of gradient compression.
            Currently `type` can be only `2bit`
            Other keys in this dictionary are optional and specific to the type
            of gradient compression.
=cut

method set_gradient_compression(HashRef[Str] $compression_params)
{
    if($self->type =~ /(?:device|dist)/)
    {
        check_call(
            AI::MXNetCAPI::KVStoreSetGradientCompression(
                $self->handle,
                scalar(keys %$compression_params),
                $compression_params
            )
        );
    }
    else
    {
        confess('Gradient compression is not supported for this type of kvstore');
    }
}

=head2  set_optimizer

    Register an optimizer to the store

    If there are multiple machines, this process (should be a worker node)
    will pack this optimizer and send it to all servers. It returns after
    this action is done.

    Parameters
    ----------
    $optimizer : AI::MXNet::Optimizer
        the optimizer
=cut

method set_optimizer(AI::MXNet::Optimizer $optimizer)
{
    my $is_worker = check_call(AI::MXNetCAPI::KVStoreIsWorkerNode());
    if($self->type =~ /dist/ and $is_worker)
    {
        my $optim_str = MIME::Base64::encode_base64(Storable::freeze($optimizer), "");
        $self->_send_command_to_servers(0, $optim_str);
    }
    else
    {
        $self->_updater(AI::MXNet::Optimizer->get_updater($optimizer));
        $self->_set_updater($self->_updater);
    }
}

=head2  type

    Get the type of this kvstore

    Returns
    -------
    $type : Str
        the string type
=cut

method type()
{
    return scalar(check_call(AI::MXNetCAPI::KVStoreGetType($self->handle)));
}

=head2  rank

    Get the rank of this worker node

    Returns
    -------
    $rank : Int
        The rank of this node, which is in [0, get_num_workers())
=cut

method rank()
{
    return scalar(check_call(AI::MXNetCAPI::KVStoreGetRank($self->handle)));
}

=head2  num_workers

    Get the number of worker nodes

    Returns
    -------
    $size : Int
        The number of worker nodes
=cut

method num_workers()
{
    return scalar(check_call(AI::MXNetCAPI::KVStoreGetGroupSize($self->handle)));
}

=head2 save_optimizer_states

    Save optimizer (updater) state to file

    Parameters
    ----------
    $fname : Str
        Path to output states file.
    :$dump_optimizer=0 : Bool, default False
            Whether to also save the optimizer itself. This would also save optimizer
            information such as learning rate and weight decay schedules.
=cut

method save_optimizer_states(Str $fname, Bool :$dump_optimizer=0)
{
    confess("Cannot save states for distributed training")
        unless defined $self->_updater;
    open(F, ">:raw", "$fname") or confess("can't open $fname for writing: $!");
    print F $self->_updater->get_states($dump_optimizer);
    close(F);
}

=head2 load_optimizer_states

    Load optimizer (updater) state from file.

    Parameters
    ----------
    $fname : Str
        Path to input states file.
=cut

method load_optimizer_states(Str $fname)
{
    confess("Cannot save states for distributed training")
        unless defined $self->_updater;
    open(F, "<:raw", "$fname") or confess("can't open $fname for reading: $!");
    my $data;
    { local($/) = undef; $data = <F>; }
    close(F);
    $self->_updater->set_states($data);
}

=head2 _set_updater

    Set a push updater into the store.

    This function only changes the local store. Use set_optimizer for
    multi-machines.

    Parameters
    ----------
    $updater : Undater
        the updater function

    Examples
    --------
    >>> my $update = sub { my ($key, input, stored) = @_;
        ...     print "update on key: $key\n";
        ...     $stored += $input * 2; };
        >>> $kv->_set_updater($update)
        >>> $kv->pull(3, out=>$a)
        >>> print $a->aspdl()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> $kv->push(3, mx->nd->ones($shape))
        update on key: 3
        >>> $kv->pull(3, out=>$a)
        >>> print $a->aspdl()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
=cut

method _set_updater(Updater $updater_func)
{
    check_call(
        AI::MXNetCAPI::KVStoreSetUpdater(
            $self->handle,
            sub {
                my ($index, $input_handle, $storage_handle) = @_;
                $updater_func->(
                    $index,
                    AI::MXNet::NDArray->_ndarray_cls($input_handle),
                    AI::MXNet::NDArray->_ndarray_cls($storage_handle)
                );
            }
        )
    );
}

=head2 _barrier

    Global barrier between all worker nodes.

    For example, assume there are n machines, we want to let machine 0 first
    init the values, and then pull the inited value to all machines. Before
    pulling, we can place a barrier to guarantee that the initialization is
    finished.
=cut

method _barrier()
{
    check_call(AI::MXNetCAPI::KVStoreBarrier($self->handle));
}

=head2 _send_command_to_servers

    Send a command to all server nodes
    Send a command to all server nodes, which will make each server node run
    KVStoreServer.controller
    This function returns after the command has been executed in all server
    nodes.

    Parameters
    ----------
    $head : Int
        the head of the command
    $body : Str
        the body of the command
=cut

method _send_command_to_servers(Int $head, Str $body)
{
    check_call(
        AI::MXNetCAPI::KVStoreSendCommmandToServers(
            $self->handle,
            $head,
            $body
        )
    );
}

=head2 create

    Create a new KVStore.

    Parameters
    ----------
    $name='local' : Str
    The type of KVStore
        - local works for multiple devices on a single machine (single process)
        - dist works for multi-machines (multiple processes)
    Returns
    -------
    kv : KVStore
        The created AI::MXNet::KVStore
=cut

method create(Str $name='local')
{
    my $handle = check_call(AI::MXNetCAPI::KVStoreCreate($name));
    return __PACKAGE__->new(handle => $handle);
}

sub _key_value
{
    my ($keys, $vals) = @_;
    if(not ref $keys)
    {
        if(blessed $vals)
        {
            return ([$keys], [$vals->handle]);
        }
        else
        {
            for my $value (@{ $vals })
            {
                assert(blessed($value) and $value->isa('AI::MXNet::NDArray'));
                return ([($keys)x@$vals], [map { $_->handle } @$vals]);
            }
        }
    }
    else
    {
        assert(not blessed($vals) and @$keys == @$vals);
        my @c_keys;
        my @c_vals;
        for(zip($keys, $vals)) {
            my ($key, $val) = @$_;
            my ($c_key, $c_val) = _key_value($key, $val);
            push @c_keys, @$c_key;
            push @c_vals, @$c_val;
        }
        return (\@c_keys, \@c_vals);
    }
}

1;
