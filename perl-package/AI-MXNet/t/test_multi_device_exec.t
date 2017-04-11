use strict;
use warnings;
use Test::More tests => 10;
use AI::MXNet qw(mx);
use AI::MXNet::Base;

sub test_ctx_group
{
    my ($data, $fc1, $act1);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage1');
        $data = mx->symbol->Variable('data');
        $fc1  = mx->symbol->FullyConnected(data => $data, name=>'fc1', num_hidden=>128);
        $act1 = mx->symbol->Activation(data => $fc1, name=>'relu1', act_type=>"relu");
    }
    my %set_stage1 = map { $_ => 1 } @{ $act1->list_arguments };

    my ($fc2, $act2, $fc3, $mlp);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage2');
        $fc2  = mx->symbol->FullyConnected(data => $act1, name => 'fc2', num_hidden => 64);
        $act2 = mx->symbol->Activation(data => $fc2, name=>'relu2', act_type=>"relu");
        $fc3  = mx->symbol->FullyConnected(data => $act2, name=>'fc3', num_hidden=>10);
        $fc3  = mx->symbol->BatchNorm($fc3);
        $mlp  = mx->symbol->SoftmaxOutput(data => $fc3, name => 'softmax');
    }
    my %set_stage2 = map { $_ => 1 } @{ $mlp->list_arguments };
    for my $k (keys %set_stage1)
    {
        delete $set_stage2{$k};
    }

    my $group2ctx = {
        stage1 => mx->cpu(1),
        stage2 => mx->cpu(2)
    };

    my $texec = $mlp->simple_bind(
        ctx       => mx->cpu(0),
        group2ctx => $group2ctx,
        shapes    => { data => [1,200] }
    );

    zip(sub {
        my ($arr, $name) = @_;
        if(exists $set_stage1{ $name })
        {
            ok($arr->context == $group2ctx->{stage1});
        }
        else
        {
            ok($arr->context == $group2ctx->{stage2});
        }
    }, $texec->arg_arrays, $mlp->list_arguments());
}

test_ctx_group();
