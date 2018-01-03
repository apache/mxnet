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

package AI::MXNet::Gluon::Utils;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Digest::SHA qw(sha1_hex);
use File::Path qw(make_path);
use HTTP::Tiny;
use Exporter;
use base qw(Exporter);
@AI::MXNet::Gluon::Utils::EXPORT_OK = qw(download);

=head1 NAME

    AI::MXNet::Gluon::Utils
=cut

=head1 DESCRIPTION

    Miscellaneous utilities.
=cut

=head2 split_data

    Splits an NDArray into `num_slice` slices along `batch_axis`.
    Usually used for data parallelism where each slices is sent
    to one device (i.e. GPU).

    Parameters
    ----------
    $data : NDArray
        A batch of data.
    $num_slice : int
        Number of desired slices.
    $batch_axis=0 : int, default 0
        The axis along which to slice.
    :$even_split=1 : bool, default True
        Whether to force all slices to have the same number of elements.
        If `True`, an error will be raised when `num_slice` does not evenly
        divide `data.shape[batch_axis]`.

    Returns
    -------
    array ref of NDArray
        Return value is a array ref even if `num_slice` is 1.
=cut


method split_data(AI::MXNet::NDArray $data, Int $num_slice, Int $batch_axis=0, Bool :$even_split=1)
{
    my $size = $data->shape->[$batch_axis];
    if($size < $num_slice)
    {
        Carp::confess(
            sprintf(
                "Too many slices for data with shape (%s). Arguments are ".
                "num_slice=%d and batch_axis=%d.",
                join(',', @{ $data->shape }), $num_slice, $batch_axis
            )
        );
    }
    if($even_split and $size % $num_slice != 0)
    {
        Carp::confess(
            sprintf(
                "data with shape %s cannot be evenly split into %d slices along axis %d. ".
                "Use a batch size that's multiple of %d or set even_split=False to allow ".
                "uneven partitioning of data.",
                join(',', @{ $data->shape }), $num_slice, $batch_axis, $num_slice
            )
        );
    }
    my $step = int($size/$num_slice);
    my $slices = [];
    if($batch_axis == 0)
    {
        for my $i (0 .. $num_slice-1)
        {
            if($i < $num_slice-1)
            {
                push @$slices, $data->slice([$i*$step, ($i+1)*$step-1]);
            }
            else
            {
                push @$slices, $data->slice([$i*$step, $size-1]);
            }
        }
    }
    elsif($even_split)
    {
        $slices = AI::MXNet::NDArray->split($data, num_outputs => $num_slice, axis => $batch_axis);
    }
    else
    {
        for my $i (0 .. $num_slice-1)
        {
            if($i < $num_slice-1)
            {
                push @$slices, $data->slice_axis($batch_axis, $i*$step, ($i+1)*$step);
            }
            else
            {
                push @$slices, $data->slice_axis($batch_axis, $i*$step, $size);
            }
        }
    }
    return $slices;
}

=head2 split_and_load

    Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.

    Parameters
    ----------
    $data : AcceptableInput
        A batch of data.
    :$ctx_list : list of Context
        A list of Contexts.
    :$batch_axis : int, default 0
        The axis along which to slice.
    :$even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArray
        Each corresponds to a context in `ctx_list`.
=cut

method split_and_load(
    PDL|PDL::Matrix|ArrayRef|AI::MXNet::NDArray $data,
    ArrayRef[AI::MXNet::Context] :$ctx_list,
    Int :$batch_axis=0,
    Bool :$even_split=1
)
{
    if(not (blessed $data and $data->isa('AI::MXNet::NDArray')))
    {
        $data = AI::MXNet::NDArray->array($data, ctx => $ctx_list->[0])
    }
    if(@{ $ctx_list } == 1)
    {
        return [$data->as_in_context($ctx_list->[0])];
    }
    my $slices = __PACKAGE__->split_data($data, scalar(@$ctx_list), $batch_axis, $even_split);
    my @ret;
    for(zip($slices, $ctx_list)) {
        my ($i, $ctx) = @$_;
        push @ret, $i->as_in_context($ctx);
    }
    return \@ret;
}

=head2 clip_global_norm

    Rescales NDArrays so that the sum of their 2-norm is smaller than `max_norm`.
=cut

method clip_global_norm(ArrayRef[AI::MXNet::NDArray] $arrays, Num $max_norm)
{
    assert(@$arrays > 0);
    my $total_norm = 0;
    for my $arr (@$arrays)
    {
        $arr = $arr->reshape([-1]);
        $total_norm += AI::MXNet::NDArray->dot($arr, $arr);
    }
    $total_norm = sqrt($total_norm->asscalar);
    my $scale = $max_norm / ($total_norm + 1e-8);
    if($scale < 1)
    {
        $_ *= $scale for @{ $arrays };
    }
    return $total_norm
}

=head2 check_sha1

    Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
=cut

func check_sha1(Str $filename, Str $sha1_hash)
{
    local($/) = undef;
    open(F, $filename) or Carp::confess("can't open $filename $!");
    my $data = <F>;
    close(F);
    return sha1_hex($data) eq $sha1_hash;
}

=head2 download

    Download an given URL

    Parameters
    ----------
    $url : str
        URL to download
    :$path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    :$overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    :$sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
=cut

func download(Str $url, Maybe[Str] :$path=, Bool :$overwrite=0, Maybe[Str] :$sha1_hash=)
{
    my $fname;
    $path =~ s/~/$ENV{HOME}/ if defined $path;
    if(not defined $path)
    {
        $fname = (split(m[/], $url))[-1];
    }
    elsif(-d $path)
    {
        $fname = join('/', $path, (split(m[/], $url))[-1]);
    }
    else
    {
        $fname = $path;
    }
    if($overwrite or not -f $fname or ($sha1_hash and not check_sha1($fname, $sha1_hash)))
    {
        $fname =~ s/~/$ENV{HOME}/;
        my $dirname = $fname;
        $dirname =~ s/[^\/]+$//;
        if(not -d $dirname)
        {
            make_path($dirname);
        }
        warn "Downloading $fname from $url ...\n";
        my $response = HTTP::Tiny->new->get($url);
        Carp::confess("download of url failed! ($response->{status} $response->{reason})\n")
            unless $response->{success};
        open(F, ">$fname") or Carp::confess("can't open $fname: $!");
        print F $response->{content};
        close(F);
    }
    return $fname
}

1;
