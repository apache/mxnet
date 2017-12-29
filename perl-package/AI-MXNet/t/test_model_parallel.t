use strict;
use warnings;
use Test::More tests => 4;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(reldiff);
use AI::MXNet::Base;

sub test_chain
{
    my $ctx1 = mx->cpu(0);
    my $ctx2 = mx->cpu(1);
    my $n = 2;
    my $data1 = mx->sym->Variable('data1');
    my $data2 = mx->sym->Variable('data2');
    my $data3 = mx->sym->Variable('data2');
    my $net;
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'dev1');
        $net = $data1 + $data2;
        $net = $net * 3;
    }
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'dev2');
        $net = $net + $data3;
    }

    my $arr = [];
    my $arr_grad = [];
    my $shape = [4, 5];
    {
        local($mx::Context) = $ctx1;
        for (0..$n-1)
        {
            push @$arr, mx->nd->empty($shape);
            push @$arr_grad, mx->nd->empty($shape);
        }
    }
    {
        local($mx::Context) = $ctx2;
        push @$arr, mx->nd->empty($shape);
        push @$arr_grad, mx->nd->empty($shape);
    }

    my $exec1 = $net->bind(
        ctx          => $ctx1,
        args         => $arr,
        args_grad    => $arr_grad,
        group2ctx    => { dev1 => $ctx1, dev2 => $ctx2 }
    );
    $arr->[0] .= 1;
    $arr->[1] .= 2;
    $arr->[2] .= 3;
    my $arr2 = [map { $_->copyto($ctx1) } @$arr];
    my $arr_grad2 = [map { $_->copyto($ctx1) } @$arr_grad];
    my $exec2 = $net->bind(
        ctx       => $ctx1,
        args      => $arr2,
        args_grad => $arr_grad2
    );

    $exec1->forward(1);
    $exec2->forward(1);
    ok(reldiff($exec1->outputs->[0]->aspdl, $exec2->outputs->[0]->aspdl) < 1e-6);
    my $out_grad = mx->nd->empty($shape, ctx => $ctx1);
    $out_grad .= 1;
    $exec1->backward([$out_grad]);
    $exec2->backward([$out_grad->copyto($ctx1)]);
    for(zip($arr_grad, $arr_grad2)) {
        my ($a, $b) = @$_;
        ok(reldiff($a->aspdl, $b->aspdl) < 1e-6);
    }
}

test_chain();
