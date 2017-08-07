use strict;
use warnings;
use Test::More tests => 18;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(mlp2);

sub _test_shapes
{
    my ($sym, $arg_shapes, %expected_shapes) = @_;
    my %arg_shape_dict;
    @arg_shape_dict{ @{ $sym->list_arguments() } } = @{ $arg_shapes };
    while(my ($k, $v) = each %expected_shapes)
    {
        is_deeply($arg_shape_dict{$k}, $v);
    }
}

sub test_mlp2_infer_shape
{
    # Build MLP
    my $out = mlp2();
    # infer shape
    my $data_shape = [100, 100];
    my($arg_shapes, $out_shapes, $aux_shapes) = $out->infer_shape(data=>$data_shape);
    ok(@$out_shapes == 1);
    is_deeply($out_shapes->[0], [100, 10]);
    my %true_shapes = (
        fc2_bias   => [10],
        fc2_weight => [10, 1000],
        fc1_bias   => [1000],
        fc1_weight => [1000,100]
    );
    _test_shapes($out, $arg_shapes, %true_shapes);
}

sub test_mlp2_infer_error
{
    # Test shape inconsistent case
    my $out = mlp2();
    my $weight_shape = [1, 100];
    my $data_shape   = [100, 100];
    eval { $out->infer_shape(data=>$data_shape, fc1_weight=>$weight_shape) };
    like($@, qr/Shape inconsistent/);
}

sub test_backward_infer
{
    my $w = mx->sym->Variable("weight");
    my $wshift = mx->sym->Variable("wshift", shape=>[1]);
    my $data = mx->sym->Variable("data");
    # broadcast add here, not being able to deduce shape correctly
    my $wt = mx->sym->broadcast_add($w, $wshift);
    # shape constraint, this is what enables backward shape inference
    $wt = mx->sym->_identity_with_attr_like_rhs($wt, $w);
    my $net = mx->sym->FullyConnected(data=>$data, weight=>$wt, num_hidden=>11, no_bias=>1);
    my $data_shape = [7, 100];
    my ($arg_shapes, $out_shapes, $aux_shapes) = $net->infer_shape(data=>$data_shape);
    _test_shapes($net, $arg_shapes, weight=>[11,100]);
}

sub test_incomplete_infer_elewise
{
    my $a = mx->sym->Variable('a', shape=>[0, 10]);
    my $b = mx->sym->Variable('b', shape=>[12, 0]);
    my $c = $a + $b;
    my ($arg_shapes) = $c->infer_shape();
    _test_shapes($c, $arg_shapes, a=>[12,10], b=>[12,10]);
}

sub test_incomplete_infer_mlp
{
    my $a = mx->sym->Variable('a', shape=>[0, 10]);
    my $b = mx->sym->FullyConnected(data=>$a, num_hidden=>21);
    my $c = mx->sym->Variable('c', shape=>[5, 0]);
    my $d = $b + $c;
    my ($arg_shapes) = $d->infer_shape();
    _test_shapes($d, $arg_shapes, a=>[5,10], c=>[5,21]);
}

sub test_incomplete_infer_slicechannel
{
    my $a = mx->sym->Variable('a', shape=>[0, 10]);
    my $b = mx->sym->SliceChannel(data=>$a, num_outputs=>10, axis=>1, squeeze_axis=>1);
    my $c = mx->sym->Variable('c', shape=>[5]);
    my $d = @{$b}[1] + $c;
    my ($arg_shapes) = $d->infer_shape();
    _test_shapes($d, $arg_shapes, a=>[5,10]);

    $a = mx->sym->Variable('a', shape=>[0, 15, 0]);
    $b = mx->sym->SliceChannel(data=>$a, num_outputs=>3, squeeze_axis=>0);
    $c = mx->sym->Variable('c', shape=>[3, 5, 2]);
    $d = @{$b}[1] + $c;
    ($arg_shapes) = $d->infer_shape();
    _test_shapes($d, $arg_shapes, a=>[3,15,2]);
}

sub test_incomplete_infer_convolution
{
    my $a = mx->sym->Variable('a', shape=>[0, 10, 0, 0]);
    my $b = mx->sym->Convolution(data=>$a, num_filter=>21, kernel=>[3, 3], dilate=>[1, 1], pad=>[1, 1]);
    my $c = mx->sym->Variable('c', shape=>[5, 21, 32, 32]);
    my $d = $b + $c;
    my ($arg_shapes) = $d->infer_shape();
    _test_shapes($d, $arg_shapes, a=>[5, 10, 32, 32]);
}

sub test_incomplete_infer_concat
{
    my $a = mx->sym->Variable('a', shape=>[0, 10]);
    my $b = mx->sym->Variable('b', shape=>[0, 5]);
    my $c = mx->sym->Concat($a, $b, num_args=>2, dim=>1);
    my $d = mx->sym->Variable('d', shape=>[2, 0]);
    $d = $d + $c;
    my ($arg_shapes) = $d->infer_shape();
    _test_shapes($d, $arg_shapes, a=>[2,10], b=>[2,5], d=>[2,15]);
}

test_mlp2_infer_shape();
test_mlp2_infer_error();
test_backward_infer();
test_incomplete_infer_elewise();
test_incomplete_infer_mlp();
test_incomplete_infer_slicechannel();
test_incomplete_infer_convolution();
test_incomplete_infer_concat();
