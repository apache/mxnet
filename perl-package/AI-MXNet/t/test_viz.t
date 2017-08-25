use AI::MXNet qw(mx);
use Test::More tests => 1;

sub test_print_summary
{
    my $data = mx->sym->Variable('data');
    my $bias = mx->sym->Variable('fc1_bias', lr_mult => 1.0);
    my $conv1= mx->sym->Convolution(data => $data, name => 'conv1', num_filter => 32, kernel => [3,3], stride => [2,2]);
    my $bn1  = mx->sym->BatchNorm(data => $conv1, name => "bn1");
    my $act1 = mx->sym->Activation(data => $bn1, name => 'relu1', act_type => "relu");
    my $mp1  = mx->sym->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride => [2,2], pool_type => 'max');
    my $fc1  = mx->sym->FullyConnected(data => $mp1, bias => $bias, name => 'fc1', num_hidden => 10, lr_mult => 0);
    my $fc2  = mx->sym->FullyConnected(data => $fc1, name => 'fc2', num_hidden => 10, wd_mult => 0.5);
    mx->viz->print_summary($fc2);
    my $shape = { data => [1,3,28,28] };
    mx->viz->print_summary($fc2, $shape);
}

test_print_summary();
ok(1);
