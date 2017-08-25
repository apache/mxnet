use AI::MXNet qw(mx);
use Test::More tests => 31;
use AI::MXNet::TestUtils qw(same reldiff GetMNIST_ubyte GetCifar10);
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
test_MNISTIter();
test_Cifar10Rec();
