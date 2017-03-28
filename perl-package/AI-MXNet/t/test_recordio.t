use strict;
use warnings;
use AI::MXNet qw(mx);
use Test::More tests => 1711;
use File::Temp qw/tempfile/;
use PDL;

sub test_recordio
{
    my ($fd, $frec) = tempfile();
    my $N = 255;

    my $writer = mx->recordio->MXRecordIO($frec, 'w');
    for my $i (0..$N-1)
    {
        $writer->write(chr($i));
    }
    undef $writer;

    my $reader = mx->recordio->MXRecordIO($frec, 'r');
    for my $i (0..$N-1)
    {
        my $res = $reader->read;
        is($res, chr($i));
    }
}

sub test_indexed_recordio
{
    my ($fi, $fidx) = tempfile();
    my ($fr, $frec) = tempfile();
    my $N = 255;

    my $writer = mx->recordio->MXIndexedRecordIO($fidx, $frec, 'w');
    for my $i (0..$N-1)
    {
        $writer->write_idx($i, chr($i));
    }
    undef $writer;

    my $reader = mx->recordio->MXIndexedRecordIO($fidx, $frec, 'r');
    my @keys = @{ $reader->keys };
    is_deeply([sort {$a <=> $b} @keys], [0..$N-1]);
    @keys = List::Util::shuffle(@keys);
    for my $i (@keys)
    {
        my $res = $reader->read_idx($i);
        is($res, chr($i));
    }
}

sub test_recordio_pack_label
{
    my $N = 25;
    my @ascii_uppercase_and_digits = ('A'..'Z', 0..9);
    for my $i (1..$N-1)
    {
        for my $j (0..$N-1)
        {
            my $content = join('', map { $ascii_uppercase_and_digits[int(rand(36))] } 0..$j-1);
            my $label = mx->nd->array(random($i), dtype => 'float32')->aspdl;
            my $header = [0, $label, 0, 0];
            my $s = mx->recordio->pack($header, $content);
            my ($rheader, $rcontent) = mx->recordio->unpack($s);
            ok(($label == $rheader->label)->all);
            ok($content eq $rcontent);
        }
    }
}

test_recordio_pack_label();
test_recordio();
test_indexed_recordio();