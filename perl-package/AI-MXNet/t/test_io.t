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

use AI::MXNet qw(mx);
use Test::More tests => 36;
use AI::MXNet::TestUtils qw(same reldiff almost_equal GetMNIST_ubyte GetCifar10 randint rand_sparse_ndarray dies_ok);
use PDL;
use PDL::Types;
use PDL::NiceSlice;
$|++;


sub test_Cifar10Rec()
{
    GetCifar10();
    my $dataiter = mx->io->ImageRecordIter({
            path_imgrec => "data/cifar/train.rec",
            mean_img => "data/cifar/cifar10_mean.bin",
            rand_crop => 0,
            and_mirror => 0,
            shuffle => 0,
            data_shape => [3,28,28],
            batch_size => 100,
            preprocess_threads => 4,
            prefetch_buffer => 1
    });
    my @labelcount;
    my $batchcount = 0;
    while(my $batch = <$dataiter>)
    {
        my $nplabel = $batch->label->[0];
        for my $i (0..$nplabel->shape->[0]-1)
        {
            $labelcount[int($nplabel->at($i)->asscalar)] += 1;
        }
    }
    for my $i (0..9)
    {
        ok($labelcount[$i] == 5000);
    }
}

sub test_NDArrayIter()
{
    my $datas  = ones(PDL::Type->new(6), 2, 2, 1000);
    my $labels = ones(PDL::Type->new(6), 1, 1000);
    for my $i (0..999)
    {
        $datas(:,:,$i) .= $i / 100;
        $labels(:,$i) .= $i / 100;
    }
    my $dataiter = mx->io->NDArrayIter(
        data => $datas,
        label => $labels,
        batch_size => 128,
        shuffle => 1,
        last_batch_handle => 'pad'
    );
    my $batchidx = 0;
    while(<$dataiter>)
    {
        $batchidx += 1;
    }
    is($batchidx, 8);
    $dataiter = mx->io->NDArrayIter(
        data => $datas,
        label => $labels,
        batch_size => 128,
        shuffle => 0,
        last_batch_handle => 'pad'
    );
    $batchidx = 0;
    my @labelcount;
    my $i = 0;
    for my $batch (@{ $dataiter })
    {
        my $label = $batch->label->[0];
        my $flabel = $label->aspdl->flat;
        ok($batch->data->[0]->aspdl->slice(0,0,'X')->flat->at(0) == $flabel->at(0));
        for my $i (0..$label->shape->[0]-1)
        {
            $labelcount[$flabel->at($i)] += 1;
        }
    }
    for my $i (0..9)
    {
        if($i == 0)
        {
            ok($labelcount[$i] == 124);
        }
        else
        {
            ok($labelcount[$i] == 100);
        }
    }
}

sub test_NDArrayIter_csr
{
    # creating toy data
    my $num_rows = 20;
    my $num_cols = 20;
    my $batch_size = 6;
    my $shape = [$num_rows, $num_cols];
    my ($csr) = rand_sparse_ndarray($shape, 'csr', density => 0.5);
    my $dns = $csr->aspdl;
    dies_ok(sub { mx->io->NDArrayIter->new(data => $csr, label => $dns, batch_size => $batch_size) });

    # AI::MXNet::NDArray::CSR with shuffle
    my $csr_iter = mx->io->NDArrayIter(
            data => Hash::Ordered->new(csr_data => $csr, dns_data => $dns),
            label => $dns,
            batch_size => $batch_size,
            shuffle=>1, last_batch_handle=>'discard'
    );
    my $num_batch = 0;
    for my $batch (@{ $csr_iter })
    {
        $num_batch += 1;
    }

    ok($num_batch == int($num_rows / $batch_size));

    # make iterators
    $csr_iter = mx->io->NDArrayIter(data => $dns, label => $dns, batch_size => $batch_size, last_batch_handle=>'discard');
    my $begin = 0;
    for my $batch (@{ $csr_iter })
    {
        my $expected = mx->nd->zeros([$batch_size, $num_cols])->aspdl;
        my $end = $begin + $batch_size;
        $expected->slice('X', [0, $batch_size-1]) .= $dns->slice('X', [$begin, $end-1]);
        if($end > $num_rows)
        {
            $expected->slice('X', [0, $end - $num_rows - 1]) .= $dns->slice('X', [0, $end - $num_rows - 1]);
        }
        ok(almost_equal($batch->data->[0]->aspdl, $expected));
        $begin += $batch_size;
    }
}

sub test_MNISTIter()
{
    GetMNIST_ubyte();

    my $batch_size = 100;
    my $train_dataiter = mx->io->MNISTIter({
            image => "data/train-images-idx3-ubyte",
            label => "data/train-labels-idx1-ubyte",
            data_shape => [784],
            batch_size => $batch_size,
            shuffle => 1,
            flat => 1,
            silent => 0,
            seed => 10
    });
    # test_loop
    my $nbatch = 60000 / $batch_size;
    my $batch_count = 0;
    for my $batch (@{ $train_dataiter})
    {
        $batch_count += 1;
    }
    ok($nbatch == $batch_count);
    # test_reset
    $train_dataiter->reset();
    $train_dataiter->iter_next();
    my $label_0 = $train_dataiter->getlabel->aspdl->flat;
    $train_dataiter->iter_next;
    $train_dataiter->iter_next;
    $train_dataiter->iter_next;
    $train_dataiter->reset;
    $train_dataiter->iter_next;
    my $label_1 = $train_dataiter->getlabel->aspdl->flat;
    ok(sum($label_0 - $label_1) == 0);
}

test_NDArrayIter();
test_NDArrayIter_csr();
test_MNISTIter();
test_Cifar10Rec();
