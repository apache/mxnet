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

package AI::MXNet::Gluon::Data::Vision::DownloadedDataSet;
use strict;
use warnings;
use File::Path qw(make_path);
use Archive::Tar;
use IO::Zlib;
use IO::File;
use Mouse;
use AI::MXNet::Function::Parameters;
has 'root'           => (is => 'ro', isa => 'Str', required => 1);
has 'train'          => (is => 'ro', isa => 'Bool', required => 1);
has 'transform'      => (is => 'ro', isa => 'Maybe[CodeRef]');
has [qw(data label)] => (is => 'rw', init_arg => undef);
extends 'AI::MXNet::Gluon::Data::Set';
method python_constructor_arguments() { ['root', 'train', 'transform'] }

sub BUILD
{
    my $self = shift;
    my $root = $self->root;
    $root =~ s/~/$ENV{HOME}/;
    if(not -d $root)
    {
        make_path($root);
    }
    $self->_get_data;
}

method at(Index $idx)
{
    if(defined $self->transform)
    {
        return [$self->transform->($self->data->at($idx), $self->label->at($idx))];
    }
    return [$self->data->at($idx), $self->label->at($idx)];
}

method len() { $self->label->len }
method _get_data() { confess("Not Implemented") }

package AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::MNIST;
use Mouse;
use AI::MXNet::Gluon::Utils qw(download);
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::Data::Vision::DownloadedDataSet';

=head1 NAME

    AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::MNIST
=cut

=head1 DESCRIPTION

    MNIST handwritten digits dataset from `http://yann.lecun.com/exdb/mnist`_.

    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
        Defaults to ~/.mxnet/datasets/mnist
    train : bool
        Whether to load the training or testing set.
        Defaults to True
    transform : function
        A user defined callback that transforms each instance. For example

    transform => sub { my ($data, $label) = @_; return ($data->astype('float32')/255, $label) }
=cut

has [qw/_base_url _train_data _train_label _test_data _test_label/] => (is => 'rw');
has '+root'  => (default => '~/.mxnet/datasets/mnist');
has '+train' => (default => 1);
has '_base_url'    => (is => 'ro', default => 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/');
has '_train_data'  => (is => 'ro', default => sub { ['train-images-idx3-ubyte.gz',
                                                     '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d'] });
has '_train_label' => (is => 'ro', default => sub { ['train-labels-idx1-ubyte.gz',
                                                     '2a80914081dc54586dbdf242f9805a6b8d2a15fc'] });
has '_test_data'   => (is => 'ro', default => sub { ['t10k-images-idx3-ubyte.gz',
                                                     'c3a25af1f52dad7f726cce8cacb138654b760d48'] });
has '_test_label'  => (is => 'ro', default => sub { ['t10k-labels-idx1-ubyte.gz',
                                                     '763e7fa3757d93b0cdec073cef058b2004252c17'] });

method _get_data()
{
    my ($data, $label);
    if($self->train)
    {
        ($data, $label) = ($self->_train_data, $self->_train_label);
    }
    else
    {
        ($data, $label) = ($self->_test_data, $self->_test_label);
    }
    my $data_file = download($self->_base_url . $data->[0], path => $self->root,
                             sha1_hash => $data->[1]);
    my $label_file = download($self->_base_url . $label->[0], path => $self->root,
                             sha1_hash => $label->[1]);
    my $fh = new IO::Zlib;
    my ($l, $d);
    if ($fh->open($label_file, "rb"))
    {
        $fh->read($l, 100_000_000);
        $l = substr($l, 8);
        my $p = PDL->new_from_specification(PDL::Type->new(0), length($l));
        ${$p->get_dataref} = $l;
        $p->upd_data;
        $l = $p;
        $fh->close;
        $l = AI::MXNet::NDArray->array($l, dtype => 'int32')->aspdl;
    }
    if ($fh->open($data_file, "rb"))
    {
        $fh->read($d, 100_000_000);
        $d = substr($d, 16);
        my $p = PDL->new_from_specification(PDL::Type->new(0), length($d));
        ${$p->get_dataref} = $d;
        $p->upd_data;
        $d = $p;
        $fh->close;
        $d->reshape(1, 28, 28, $l->dim(-1));
    }
    $self->data(AI::MXNet::NDArray->array($d, dtype => 'uint8'));
    $self->label($l);
}

__PACKAGE__->register('AI::MXNet::Gluon::Data::Vision');

package AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::FashionMNIST;
use Mouse;

=head1 NAME

    AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::FashionMNIST
=cut

=head1 DESCRIPTION

    A dataset of Zalando's article images consisting of fashion products,
    a drop-in replacement of the original MNIST dataset from
    `https://github.com/zalandoresearch/fashion-mnist`_.

    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
        Defaults to ~/.mxnet/datasets/mnist
    train : bool
        Whether to load the training or testing set.
        Defaults to True
    transform : function
        A user defined callback that transforms each instance. For example

    transform => sub { my ($data, $label) = @_; return ($data->astype('float32')/255, $label) }
=cut

extends 'AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::MNIST';
has '+root'         => (default => '~/.mxnet/datasets/fashion-mnist');
has '+_base_url'    => (default => 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/');
has '+_train_data'  => (default => sub { ['train-images-idx3-ubyte.gz',
                                          '0cf37b0d40ed5169c6b3aba31069a9770ac9043d'] });
has '+_train_label' => (default => sub { ['train-labels-idx1-ubyte.gz',
                                          '236021d52f1e40852b06a4c3008d8de8aef1e40b'] });
has '+_test_data'   => (default => sub { ['t10k-images-idx3-ubyte.gz',
                                          '626ed6a7c06dd17c0eec72fa3be1740f146a2863'] });
has '+_test_label'  => (default => sub { ['t10k-labels-idx1-ubyte.gz',
                                          '17f9ab60e7257a1620f4ad76bbbaf857c3920701'] });

__PACKAGE__->register('AI::MXNet::Gluon::Data::Vision');

package AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::CIFAR10;
use Mouse;
use AI::MXNet::Gluon::Utils qw(download);
use AI::MXNet::Base;
use Cwd;
extends 'AI::MXNet::Gluon::Data::Vision::DownloadedDataSet';

=head1 NAME

    AI::MXNet::Gluon::Data::Vision::DownloadedDataSet::CIFAR10
=cut

=head1 DESCRIPTION

    CIFAR10 image classification dataset from `https://www.cs.toronto.edu/~kriz/cifar.html`_.

    Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
    train : bool
        Whether to load the training or testing set.
    transform : function
        A user defined callback that transforms each instance. For example:

    transform => sub { my ($data, $label) = @_; return ($data->astype('float32')/255, $label) }
=cut
has '+root'  => (default => '~/.mxnet/datasets/cifar10');
has '+train' => (default => 1);
has '_file_hashes' => (is => 'ro', default => sub { +{
    qw/data_batch_1.bin aadd24acce27caa71bf4b10992e9e7b2d74c2540
       data_batch_2.bin c0ba65cce70568cd57b4e03e9ac8d2a5367c1795
       data_batch_3.bin 1dd00a74ab1d17a6e7d73e185b69dbf31242f295
       data_batch_4.bin aab85764eb3584312d3c7f65fd2fd016e36a258e
       data_batch_5.bin 26e2849e66a845b7f1e4614ae70f4889ae604628
       test_batch.bin   67eb016db431130d61cd03c7ad570b013799c88c/
    } });

method _read_batch(Str $filename)
{
    my $data = join('', IO::File->new($filename)->getlines);
    $data = PDL->new_from_specification(PDL::Type->new(0), length($data))->reshape(3073, length($data)/3073);
    $data = AI::MXNet::NDArray->array($data, dtype => 'uint8');
    return (
        $data->slice('X', [1, -1])->sever->reshape([-1, 3, 32, 32])->transpose([0, 2, 3, 1]),
        $data->slice('X', 0)->astype('int32')
    );
}

method _get_data()
{
    my @file_paths = map { [$_, join('/', $self->root, 'cifar-10-batches-bin/', $_)] } keys %{ $self->_file_hashes };
    if(grep { not -f $_->[1] or not check_sha1($_->[1], $self->_file_hashes->{ $_->[0] }) } @file_paths)
    {
        my $filename = download(
            'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/cifar10/cifar-10-binary.tar.gz',
            path => $self->root,
            sha1_hash => 'fab780a1e191a7eda0f345501ccd62d20f7ed891'
        );
        my $tar = Archive::Tar->new($filename);
        my $cwd = cwd();
        chdir($self->root);
        $tar->extract;
        chdir($cwd);
    }
    my ($data, $label);
    if($self->train)
    {
        my (@data, @label);
        for my $i (1..5)
        {
            my $filename = join('/', $self->root, "data_batch_$i.bin");
            my ($data, $label) = $self->_read_batch($filename);
            push @data, $data;
            push @label, $label;
        }
        $data = AI::MXNet::NDArray->concatenate(\@data);
        $label = AI::MXNet::NDArray->concatenate(\@label);
    }
    else
    {
        my $filename = join('/', $self->root, "test_batch.bin");
        ($data, $label) = $self->_read_batch($filename);
    }
    $self->data(\@{$data});
    $self->label($label->aspdl);
}

__PACKAGE__->register('AI::MXNet::Gluon::Data::Vision');

package AI::MXNet::Gluon::Data::Vision::RecordFileSet::ImageRecordDataset;
use Mouse;
extends 'AI::MXNet::Gluon::Data::RecordFileSet';
=head1 NAME

    AI::MXNet::Gluon::Data::Vision::RecordFileSet::ImageRecordDataset
=cut

=head1 DESCRIPTION

    A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).
    transform : function
        A user defined callback that transforms each instance. For example::
=cut
has 'flag'      => (is => 'rw', isa => 'Bool', default => 1);
has 'transform' => (is => 'rw', isa => 'Maybe[CodeRef]');

method at(Int $idx)
{
    my $record = $self->SUPER::at($idx);
    my ($header, $img) = AI::MXNet::RecordIO->unpack($record);
    if(defined $self->transform)
    {
        my $data = [AI::MXNet::Image->imdecode($img)];
        return [$self->transform->(
            AI::MXNet::Image->imdecode($img, flag => $self->flag), $header->label
        )];
    }
    return [AI::MXNet::Image->imdecode($img, flag => $self->flag), $header->label];
}

__PACKAGE__->register('AI::MXNet::Gluon::Data::Vision');

package AI::MXNet::Gluon::Data::Vision::Set::ImageFolderDataset;
use Mouse;
extends 'AI::MXNet::Gluon::Data::Set';
=head1 NAME

    AI::MXNet::Gluon::Data::Vision::ImageFolderDataset
=cut

=head1 DESCRIPTION

    A dataset for loading image files stored in a folder structure like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
=cut
has 'root'      => (is => 'rw', isa => 'Str');
has 'flag'      => (is => 'rw', isa => 'Bool', default => 1);
has 'transform' => (is => 'rw', isa => 'Maybe[CodeRef]');
has [qw/exts
    synsets
    items/]     => (is => 'rw', init_arg => undef);
method python_constructor_arguments() { ['root', 'flag', 'transform'] }

sub BUILD
{
    my $self = shift;
    my $root = $self->root;
    $root =~ s/~/$ENV{HOME}/;
    $self->root($root);
    $self->exts({'.jpg', 1, '.jpeg', 1, '.png', 1});
    $self->list_images($self->root);
}

method list_images(Str $root)
{
    $self->synsets([]);
    $self->items([]);

    for my $path (sort(glob("$root/*")))
    {
        my $folder = $path;
        $folder =~ s,^.+/,,;
        if(not -d $path)
        {
            AI::MXNet::Logging->warning("Ignoring %s, which is not a directory.", $folder);
            next;
        }
        my $label = @{ $self->synsets };
        push @{ $self->synsets }, $folder;
        for my $filename (sort(glob("$path/*")))
        {
            my ($ext) = $filename =~ /(\.[^\.]+)$/;
            if(not $ext or not exists $self->exts->{lc $ext})
            {
                AI::MXNet::Logging->warning(
                    'Ignoring %s of type %s. Only support .jpg, .jpeg, .png',
                    $filename, $ext//'undef'
                );
                next;
            }
            push @{ $self->items }, [$filename, AI::MXNet::NDArray->array([$label], dtype => 'int32')->aspdl];
        }
    }
}

method at(Int $idx)
{
    my $img = AI::MXNet::Image->imread($self->items->[$idx][0], flag => $self->flag);
    my $label = $self->items->[$idx][1];
    if(defined $self->transform)
    {
        return [$self->transform($img, $label)];
    }
    return [$img, $label];
}

method len()
{
    return scalar(@{ $self->items });
}

__PACKAGE__->register('AI::MXNet::Gluon::Data::Vision');

1;
