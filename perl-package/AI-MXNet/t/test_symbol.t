use strict;
use warnings;
use Test::More tests => 98;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(mlp2 conv check_consistency zip assert enumerate);
use Storable qw(freeze thaw);
use PDL;

sub test_symbol_compose
{
    my $data = mx->symbol->Variable('data');
    my $net1 = mx->symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>10);
    $net1 = mx->symbol->FullyConnected(data=>$net1, name=>'fc2', num_hidden=>100);
    is_deeply($net1->list_arguments(), ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']);

    my $net2 = mx->symbol->FullyConnected(name=>'fc3', num_hidden=>10);
    $net2 = mx->symbol->Activation(data=>$net2, act_type=>'relu');
    $net2 = mx->symbol->FullyConnected(data=>$net2, name=>'fc4', num_hidden=>20);
    my $composed = &{$net2}(fc3_data=>$net1, name=>'composed');
    my $multi_out = mx->symbol->Group([$composed, $net1]);
    ok(@{ $multi_out->list_outputs() } == 2);
}

test_symbol_compose();

sub test_symbol_copy
{
    my $data = mx->symbol->Variable('data');
    my $data_2 = $data->deepcopy;
    is($data->tojson, $data_2->tojson);
}

test_symbol_copy();

sub test_symbol_internal
{
    my $data = mx->symbol->Variable('data');
    my $oldfc = mx->symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>10);
    my $net1 = mx->symbol->FullyConnected(data=>$oldfc, name=>'fc2', num_hidden=>100);
    is_deeply($net1->list_arguments, ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias']);

    my $internal = $net1->get_internals();
    my $fc1 = $internal->slice('fc1_output');
    is_deeply($fc1->list_arguments, $oldfc->list_arguments);
}

test_symbol_internal();

sub test_symbol_children
{
    my $data = mx->symbol->Variable('data');
    my $oldfc = mx->symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>10);
    my $net1 = mx->symbol->FullyConnected(data=>$oldfc, name=>'fc2', num_hidden=>100);

    is_deeply($net1->get_children()->list_outputs(), ['fc1_output', 'fc2_weight', 'fc2_bias']);
    is_deeply($net1->get_children()->get_children()->list_outputs() , ['data', 'fc1_weight', 'fc1_bias']);
    is_deeply($net1->get_children()->slice('fc2_weight')->list_arguments(), ['fc2_weight']);
    ok(not defined $net1->get_children()->slice('fc2_weight')->get_children());

    $data = mx->sym->Variable('data');
    my $sliced = mx->sym->SliceChannel($data, num_outputs=>3, name=>'slice');
    my $concat = mx->sym->Concat(@{ $sliced });

    is_deeply($concat->get_children()->list_outputs(),
        ['slice_output0', 'slice_output1', 'slice_output2']);
    is_deeply($sliced->get_children()->list_outputs(), ['data']);
}

test_symbol_children();

sub test_symbol_storable
{
    my $mlist = [mlp2(), conv()];
    my $data = freeze($mlist);
    my $mlist2 = thaw($data);
    zip(sub {
        my ($x, $y) = @_;
        is($x->tojson, $y->tojson);
    }, $mlist, $mlist2);
}

test_symbol_storable();

sub test_symbol_saveload
{
    my $sym = mlp2();
    my $fname = 'tmp_sym.json';
    $sym->save($fname);
    my $data2 = mx->symbol->load($fname);
    # save because of order
    is($sym->tojson, $data2->tojson);
    unlink $fname;
}

test_symbol_saveload();

sub test_symbol_infer_type
{
    my $data = mx->symbol->Variable('data');
    my $f32data = mx->symbol->Cast(data=>$data, dtype=>'float32');
    my $fc1 = mx->symbol->FullyConnected(data => $f32data, name=>'fc1', num_hidden=>128);
    my $mlp = mx->symbol->SoftmaxOutput(data => $fc1, name => 'softmax');

    my ($arg, $out, $aux) = $mlp->infer_type(data=>'float16');
    is_deeply($arg, [qw/float16 float32 float32 float32/]);
    is_deeply($out, ['float32']);
    is_deeply($aux, []);
}

test_symbol_infer_type();

sub test_symbol_infer_shape
{
    my $num_hidden = 128;
    my $num_dim    = 64;
    my $num_sample = 10;

    my $data = mx->symbol->Variable('data');
    my $prev = mx->symbol->Variable('prevstate');
    my $x2h  = mx->symbol->FullyConnected(data=>$data, name=>'x2h', num_hidden=>$num_hidden);
    my $h2h  = mx->symbol->FullyConnected(data=>$prev, name=>'h2h', num_hidden=>$num_hidden);

    my $out  = mx->symbol->Activation(data=>mx->sym->elemwise_add($x2h, $h2h), name=>'out', act_type=>'relu');

    # shape inference will fail because information is not available for h2h
    my @ret  = $out->infer_shape(data=>[$num_sample, $num_dim]);
    is_deeply(\@ret, [undef, undef, undef]);

    my ($arg_shapes, $out_shapes, $aux_shapes) = $out->infer_shape_partial(data=>[$num_sample, $num_dim]);
    my %arg_shapes;
    @arg_shapes{ @{ $out->list_arguments } } = @{ $arg_shapes };
    is_deeply($arg_shapes{data}, [$num_sample, $num_dim]);
    is_deeply($arg_shapes{x2h_weight}, [$num_hidden, $num_dim]);
    is_deeply($arg_shapes{h2h_weight}, []);

    # now we can do full shape inference
    my $state_shape = $out_shapes->[0];
    ($arg_shapes, $out_shapes, $aux_shapes) = $out->infer_shape(data=>[$num_sample, $num_dim], prevstate=>$state_shape);
    @arg_shapes{ @{ $out->list_arguments } } = @{ $arg_shapes };
    is_deeply($arg_shapes{data}, [$num_sample, $num_dim]);
    is_deeply($arg_shapes{x2h_weight}, [$num_hidden, $num_dim]);
    is_deeply($arg_shapes{h2h_weight}, [$num_hidden, $num_hidden]);
}

test_symbol_infer_shape();

sub test_symbol_infer_shape_var
{
    #Test specifying shape information when constructing a variable
    my $shape = [2, 3];
    my $a = mx->symbol->Variable('a', shape=>$shape);
    my $b = mx->symbol->Variable('b');
    my $c = mx->symbol->elemwise_add($a, $b);
    my ($arg_shapes, $out_shapes, $aux_shapes) = $c->infer_shape();
    is_deeply($arg_shapes->[0], $shape);
    is_deeply($arg_shapes->[1], $shape);
    is_deeply($out_shapes->[0], $shape);

    $shape = [5, 6];
    ($arg_shapes, $out_shapes, $aux_shapes) = $c->infer_shape(a=>$shape);
    is_deeply($arg_shapes->[0], $shape);
    is_deeply($arg_shapes->[1], $shape);
    is_deeply($out_shapes->[0], $shape);
}

test_symbol_infer_shape_var();

sub check_symbol_consistency
{
    my ($sym1, $sym2, $ctx) = @_;
    is_deeply($sym1->list_arguments(), $sym2->list_arguments());
    is_deeply($sym1->list_auxiliary_states(), $sym2->list_auxiliary_states());
    is_deeply($sym1->list_outputs(), $sym2->list_outputs());
    check_consistency(sym => [$sym1, $sym2], ctx_list => [$ctx, $ctx]);
}

sub test_load_000800
{
    my ($data, $weight, $fc1, $act1);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage1');
        $data = mx->symbol->Variable('data', lr_mult=>0.2);
        $weight = mx->sym->Variable('fc1_weight', lr_mult=>1.2);
        $fc1  = mx->symbol->FullyConnected(data => $data, weight=>$weight, name=>'fc1', num_hidden=>128, wd_mult=>0.3);
        $act1 = mx->symbol->Activation(data => $fc1, name=>'relu1', act_type=>"relu");
    }
    my ($fc2, $act2, $fc3, $sym1);
    {
        local($mx::AttrScope) = mx->AttrScope(ctx_group=>'stage2');
        $fc2  = mx->symbol->FullyConnected(data => $act1, name => 'fc2', num_hidden => 64, lr_mult=>0.01);
        $act2 = mx->symbol->Activation(data => $fc2, name=>'relu2', act_type=>"relu");
        $fc3  = mx->symbol->FullyConnected(data => $act2, name=>'fc3', num_hidden=>10);
        $fc3  = mx->symbol->BatchNorm($fc3, name=>'batchnorm0');
        $sym1 = mx->symbol->SoftmaxOutput(data => $fc3, name => 'softmax')
    }
    { local $/ = undef; my $json = <DATA>; open(F, ">save_000800.json"); print F $json; close(F); };
    my $sym2 = mx->sym->load('save_000800.json');
    unlink 'save_000800.json';

    my %attr1 = %{ $sym1->attr_dict };
    my %attr2 = %{ $sym2->attr_dict };
    while(my ($k, $v1) = each %attr1)
    {
        ok(exists $attr2{ $k });
        my $v2 = $attr2{$k};
        while(my ($kk, $vv1) = each %{ $v1 })
        {
            if($kk =~ /^__/ and $kk =~ /__$/)
            {
                ok(exists $v2->{$kk} and $v2->{$kk} eq $vv1);
            }
        }
    }

    check_symbol_consistency($sym1, $sym2,
        {ctx => mx->cpu(0), group2ctx =>{stage1 => mx->cpu(1), stage2 => mx->cpu(2) }, shapes => { data => [1,200] }}
    );
}

test_load_000800();

__DATA__
{
  "nodes": [
    {
      "op": "null", 
      "param": {}, 
      "name": "data", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage1", 
        "lr_mult": "0.2"
      }
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc1_weight", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage1", 
        "wd_mult": "0.3", 
        "weight_lr_mult": "1.2"
      }
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc1_bias", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage1", 
        "wd_mult": "0.3", 
        "weight_lr_mult": "1.2"
      }
    }, 
    {
      "op": "FullyConnected", 
      "param": {
        "no_bias": "False", 
        "num_hidden": "128"
      }, 
      "name": "fc1", 
      "inputs": [[0, 0], [1, 0], [2, 0]], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage1", 
        "wd_mult": "0.3", 
        "weight_lr_mult": "1.2"
      }
    }, 
    {
      "op": "Activation", 
      "param": {"act_type": "relu"}, 
      "name": "relu1", 
      "inputs": [[3, 0]], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage1"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc2_weight", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage2", 
        "lr_mult": "0.01"
      }
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc2_bias", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage2", 
        "lr_mult": "0.01"
      }
    }, 
    {
      "op": "FullyConnected", 
      "param": {
        "no_bias": "False", 
        "num_hidden": "64"
      }, 
      "name": "fc2", 
      "inputs": [[4, 0], [5, 0], [6, 0]], 
      "backward_source_id": -1, 
      "attr": {
        "ctx_group": "stage2", 
        "lr_mult": "0.01"
      }
    }, 
    {
      "op": "Activation", 
      "param": {"act_type": "relu"}, 
      "name": "relu2", 
      "inputs": [[7, 0]], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc3_weight", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "fc3_bias", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "FullyConnected", 
      "param": {
        "no_bias": "False", 
        "num_hidden": "10"
      }, 
      "name": "fc3", 
      "inputs": [[8, 0], [9, 0], [10, 0]], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "batchnorm0_gamma", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "batchnorm0_beta", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "BatchNorm", 
      "param": {
        "eps": "0.001", 
        "fix_gamma": "True", 
        "momentum": "0.9", 
        "use_global_stats": "False"
      }, 
      "name": "batchnorm0", 
      "inputs": [[11, 0], [12, 0], [13, 0]], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "null", 
      "param": {}, 
      "name": "softmax_label", 
      "inputs": [], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }, 
    {
      "op": "SoftmaxOutput", 
      "param": {
        "grad_scale": "1", 
        "ignore_label": "-1", 
        "multi_output": "False", 
        "normalization": "null", 
        "out_grad": "False", 
        "preserve_shape": "False", 
        "use_ignore": "False"
      }, 
      "name": "softmax", 
      "inputs": [[14, 0], [15, 0]], 
      "backward_source_id": -1, 
      "attr": {"ctx_group": "stage2"}
    }
  ], 
  "arg_nodes": [0, 1, 2, 5, 6, 9, 10, 12, 13, 15], 
  "heads": [[16, 0]]
}