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

use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::Utils qw(download);
use Archive::Tar;
use AI::MXNet::TestUtils qw(almost_equal);
use AI::MXNet::Base;
use File::Path qw(make_path);
use IO::File;
use Test::More tests => 52;

sub test_array_dataset
{
    my $X = mx->nd->random->uniform(shape=>[10, 20]);
    my $Y = mx->nd->random->uniform(shape=>[10]);
    my $dataset = gluon->data->ArrayDataset($X, $Y);
    my $loader = gluon->data->DataLoader($dataset, 2);
    enumerate(sub {
        my ($i, $d) = @_;
        my ($x, $y) = @$d;
        ok(almost_equal($x->aspdl, $X->slice([$i*2,($i+1)*2-1])->aspdl));
        ok(almost_equal($y->aspdl, $Y->slice([$i*2,($i+1)*2-1])->aspdl));
    }, \@{ $loader });
}

test_array_dataset();

sub prepare_record
{
    my ($copy) = @_;
    if(not -d "data/test_images")
    {
        make_path('data/test_images');
    }
    if(not -d "data/test_images/test_images")
    {
        download("http://data.mxnet.io/data/test_images.tar.gz", path => "data/test_images.tar.gz");
        my $f = Archive::Tar->new('data/test_images.tar.gz');
        chdir('data');
        $f->extract;
        chdir('..');
    }
    if(not -f 'data/test.rec')
    {
        my @imgs = glob('data/test_images/*');
        my $record = mx->recordio->MXIndexedRecordIO('data/test.idx', 'data/test.rec', 'w');
        enumerate(sub {
            my ($i, $img) = @_;
            my $str_img = join('',IO::File->new("./$img")->getlines);
            my $s = mx->recordio->pack([0, $i, $i, 0], $str_img);
            $record->write_idx($i, $s);
        }, \@imgs);
    }
    if($copy)
    {
        make_path('data/images/test_images');
        `cp  data/test_images/* data/images/test_images`;
    }
    return 'data/test.rec';
}

sub test_recordimage_dataset
{
    my $recfile = prepare_record();
    my $dataset = gluon->data->vision->ImageRecordDataset($recfile);
    my $loader = gluon->data->DataLoader($dataset, 1);
    enumerate(sub {
        my ($i, $d) = @_;
        my ($x, $y) = @$d;
        ok($x->shape->[0] == 1 and $x->shape->[3] == 3);
        ok($y->asscalar == $i);
    }, \@{ $loader });
}

test_recordimage_dataset();

sub test_sampler
{
    my $seq_sampler = gluon->data->SequentialSampler(10);
    is_deeply(\@{ $seq_sampler }, [0..9]);
    my $rand_sampler = gluon->data->RandomSampler(10);
    is_deeply([sort { $a <=> $b } @{ $rand_sampler }], [0..9]);
    my $seq_batch_keep = gluon->data->BatchSampler($seq_sampler, 3, 'keep');
    is_deeply([map { @$_ } @{ $seq_batch_keep }], [0..9]);
    my $seq_batch_discard = gluon->data->BatchSampler($seq_sampler, 3, 'discard');
    is_deeply([map { @$_ } @{ $seq_batch_discard }], [0..8]);
    my $rand_batch_keep = gluon->data->BatchSampler($rand_sampler, 3, 'keep');
    is_deeply([sort { $a <=> $b } map { @$_ } @{ $rand_batch_keep }], [0..9]);
}

test_sampler();

sub test_datasets
{
    ok(gluon->data->vision->MNIST(root=>'data/mnist')->len == 60000);
    ok(gluon->data->vision->FashionMNIST(root=>'data/fashion-mnist')->len == 60000);
    ok(gluon->data->vision->CIFAR10(root=>'data/cifar10', train=>0)->len == 10000);
}

test_datasets();

sub test_image_folder_dataset
{
    prepare_record(1);
    my $dataset = gluon->data->vision->ImageFolderDataset('data/images');
    is_deeply($dataset->synsets, ['test_images']);
    ok(@{ $dataset->items } == 16);
}

test_image_folder_dataset();
