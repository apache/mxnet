use strict;
use warnings;
use Test::More tests => 8;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(same);

sub check_with_device
{
    my ($device)     = @_;
    my ($a, $b)      = (-10, 10);
    my ($mu, $sigma) = (10, 2);
    my $shape        = [100, 100];
    mx->random->seed(128);
    my $ret1 = mx->random->normal($mu, $sigma, $shape, { ctx => $device });
    my $un1  = mx->random->uniform($a, $b, $shape, { ctx => $device });
    mx->random->seed(128);
    my $ret2 = mx->random->normal($mu, $sigma, $shape, { ctx => $device });
    my $un2  = mx->random->uniform($a, $b, $shape, { ctx => $device });
    ok(same($ret1->aspdl, $ret2->aspdl));
    ok(same($un1->aspdl, $un2->aspdl));
    ok(abs($ret1->aspdl->avg - $mu) < 0.1);
    ok(abs(($ret1->aspdl->stats)[6] - $sigma) < 0.1);
    ok(abs($un1->aspdl->avg - ($a+$b)/2) < 0.1);
}

sub check_symbolic_random
{
    my ($dev) = @_;
    my ($a, $b) = (-10, 10);
    my ($mu, $sigma) = (10, 2);
    my $shape = [100, 100];
    my $X = mx->sym->Variable("X");
    my $Y = mx->sym->uniform(low=>$a, high=>$b, shape=>$shape) + $X;
    my $x = mx->nd->zeros($shape, ctx=>$dev);
    my $xgrad = mx->nd->zeros($shape, ctx=>$dev);
    my $yexec = $Y->bind(ctx => $dev, args => {X => $x}, args_grad => {X => $xgrad});
    mx->random->seed(128);
    $yexec->forward(1);
    $yexec->backward($yexec->outputs->[0]);
    my $un1 = ($yexec->outputs->[0] - $x)->copyto($dev);
    ok(same($xgrad->aspdl, $un1->aspdl));
    mx->random->seed(128);
    $yexec->forward;
    my $un2 = ($yexec->outputs->[0] - $x)->copyto($dev);
    ok(same($un1->aspdl, $un2->aspdl));
    ok(abs($un1->aspdl->avg - ($a+$b)/2) < 0.1);
}

sub test_random
{
    check_with_device(mx->cpu);
    check_symbolic_random(mx->cpu);
}

test_random();
