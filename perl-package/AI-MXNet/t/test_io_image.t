use strict;
use warnings;
use Test::More tests => 1;
use AI::MXNet qw(mx);
use Time::HiRes qw(time);

sub run_imageiter
{
    my ($path_rec, $n, $batch_size) = @_;
    $batch_size //= 32;
    my $data = mx->img->ImageIter(
        batch_size=>$batch_size,
        data_shape=>[3, 224, 224],
        path_imgrec=>$path_rec,
        kwargs => { rand_crop=>1,
        rand_resize=>1,
        rand_mirror=>1 }
    );
    $data->reset();
    my $tic = time;
    for my $i (1..$n)
    {
        $data->next;
        mx->nd->waitall;
        warn("average speed after iteration $i is " . $batch_size*$i/(time - $tic) . " samples/sec");
    }
}

run_imageiter('data/cifar/test.rec', 20);
ok(1);