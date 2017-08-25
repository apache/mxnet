use strict;
use warnings;
use Test::More tests => 3;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(reldiff);
use AI::MXNet::Base;

sub test_chain
{
    my $n = 2;
    my $data1 = mx->sym->Variable('data1');
    my $data2 = mx->sym->Variable('data2');
    my $net;
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'dev1');
        $net = $data1 + $data2;
        $net = $net * 3;
    }

    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'dev2');
        $net = $net + $data1;
    }
    my $arr;
    my $arr_grad;
    my $shape = [4, 5];
    {
        local($mx::Context) = mx->Context(mx->cpu(0));
        $arr   = [map { mx->nd->empty($shape) } 0..$n-1];
        $arr_grad = [map { mx->nd->empty($shape) } 0..$n-1];
    }

    my $exec1 = $net->bind(
        ctx          => mx->cpu(),
        args         => $arr,
        args_grad    => $arr_grad,
        group2ctx    => { dev1 => mx->cpu(0), dev2 => mx->cpu(1) }
    );
    $arr->[0] .= 1;
    $arr->[1] .= 2;
    my $arr2 = [map { $_->copyto(mx->cpu()) } @$arr];
    my $arr_grad2 = [map { $_->copyto(mx->cpu()) } @$arr_grad];
    my $exec2 = $net->bind(
        ctx       => mx->cpu(),
        args      => $arr2,
        args_grad => $arr_grad2
    );

    $exec1->forward(1);
    $exec2->forward(1);
    ok(reldiff($exec1->outputs->[0]->aspdl, $exec2->outputs->[0]->aspdl) < 1e-6);
    my $out_grad = mx->nd->empty($shape, ctx => mx->cpu(1));
    $out_grad .= 1;
    $exec1->backward([$out_grad]);
    $exec2->backward([$out_grad->copyto(mx->cpu())]);
    zip(sub {
        my ($a, $b) = @_;
        ok(reldiff($a->aspdl, $b->aspdl) < 1e-6);
    }, $arr_grad, $arr_grad2);
}

test_chain();
