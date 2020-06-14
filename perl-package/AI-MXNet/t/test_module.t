# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

use strict;
use warnings;
use Test::More tests => 19;
use AI::MXNet qw(mx);
use AI::MXNet::Base;
use AI::MXNet::TestUtils qw(almost_equal enumerate same_array dies_like rand_ndarray);
$ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE} = 0;
$ENV{MXNET_SUBGRAPH_VERBOSE} = 0;

sub test_module_layout
{
    my $sym = mx->sym->Variable('data');
    $sym = mx->sym->Activation(data=>$sym, act_type=>'relu', __layout__=>'TNC');

    my $dshape = [3, 8, 7];
    my $mod = mx->mod->Module(
        $sym,
        data_names=>['data'],
        context=>[mx->cpu(0), mx->cpu(1)]
    );
    $mod->bind(
        data_shapes=>[mx->io->DataDesc('data', $dshape, layout=>'TNC')]
    );
    $mod->init_params();
    $mod->forward(
        mx->io->DataBatch(
            data=>[mx->nd->ones($dshape)]
        ),
        is_train => 1
    );
    $mod->backward([mx->nd->ones($dshape)]);
    is_deeply($mod->get_outputs()->[0]->shape, $dshape);

    my $hdshape = [3, 4, 7];
    for my $x (@{ $mod->get_outputs(0)->[0] })
    {
        is_deeply($x->shape, $hdshape);
    }
}

sub test_save_load
{
    my $dict_equ = sub {
        is_deeply([sort keys %$a], [sort keys %$b]);
        for my $k (keys %$a)
        {
            ok(($a->{$k}->aspdl == $b->{$k}->aspdl)->all);
        }
    };
    my $sym = mx->sym->Variable('data');
    $sym = mx->sym->FullyConnected($sym, num_hidden=>100);

    # single device
    my $mod = mx->mod->Module($sym, data_names=>['data']);
    $mod->bind(data_shapes=>[['data', [10, 10]]]);
    $mod->init_params();
    $mod->init_optimizer(optimizer_params=>{learning_rate => 0.1, momentum => 0.9});
    $mod->update();
    $mod->save_checkpoint('test', 0, 1);

    my $mod2 = mx->mod->Module->load('test', 0, 1, data_names=>['data']);
    $mod2->bind(data_shapes=>[['data', [10, 10]]]);
    $mod2->init_optimizer(optimizer_params=>{learning_rate => 0.1, momentum => 0.9});
    is($mod->_symbol->tojson(), $mod2->_symbol->tojson());
    $dict_equ->(($mod->get_params())[0], ($mod2->get_params())[0]);
    $dict_equ->($mod->_updater->states, $mod2->_updater->states);

    # multi device
    $mod = mx->mod->Module($sym, data_names=>['data'], context=>[mx->cpu(0), mx->cpu(1)]);
    $mod->bind(data_shapes=>[['data', [10, 10]]]);
    $mod->init_params();
    $mod->init_optimizer(optimizer_params=>{learning_rate => 0.1, momentum => 0.9});
    $mod->update();
    $mod->save_checkpoint('test', 0, 1);

    $mod2 = mx->mod->Module->load('test', 0, 1, data_names=>['data']);
    $mod2->bind(data_shapes=>[['data', [10, 10]]]);
    $mod2->init_optimizer(optimizer_params=>{learning_rate => 0.1, momentum => 0.9});
    is($mod->_symbol->tojson(), $mod2->_symbol->tojson());
    $dict_equ->(($mod->get_params())[0], ($mod2->get_params())[0]);
    $dict_equ->($mod->_kvstore->_updater->states, $mod2->_updater->states);
    unlink('test-0000.params');
    unlink('test-0000.states');
    unlink('test-symbol.json');
}


sub test_module_reshape
{
    my $data = mx->sym->Variable('data');
    my $sym  = mx->sym->FullyConnected($data, num_hidden=>20, name=>'fc');

    my $dshape = [7, 20];
    my $mod = mx->mod->Module($sym, data_names=>['data'], context=>[mx->cpu(0), mx->cpu(1)]);
    $mod->bind(data_shapes=>[['data', $dshape]]);
    $mod->init_params();
    $mod->init_optimizer(optimizer_params=>{learning_rate => 1});

    $mod->forward(
        mx->io->DataBatch(
            data=>[mx->nd->ones($dshape)]
        ),
        is_train => 1
    );
    $mod->backward([mx->nd->ones($dshape)]);
    $mod->update();
    is_deeply($mod->get_outputs()->[0]->shape, $dshape);
    ok((($mod->get_params())[0]{fc_bias}->aspdl == -1)->all);

    $dshape = [14, 20];
    $mod->reshape(data_shapes=>[['data', $dshape]]);
    $mod->forward(
        mx->io->DataBatch(
            data=>[mx->nd->ones($dshape)]
        ),
        is_train => 1
    );
    $mod->backward([mx->nd->ones($dshape)]);
    $mod->update();
    is_deeply($mod->get_outputs()->[0]->shape, $dshape);
    ok((($mod->get_params())[0]{fc_bias}->aspdl == -3)->all);
}


sub test_module_states
{
    my $stack = mx->rnn->SequentialRNNCell();
    for my $i (0..1)
    {
        $stack->add(mx->rnn->LSTMCell(num_hidden=>20, prefix=>"lstm_l${i}_"));
    }
    my $begin_state = $stack->begin_state(func=>mx->sym->can('Variable'));
    my (undef, $states) = $stack->unroll(10, begin_state=>$begin_state, inputs=>mx->sym->Variable('data'));

    my $state_names = [map { $_->name } @$begin_state];
    my $mod = mx->mod->Module(
        mx->sym->Group($states), context=>[mx->cpu(0), mx->cpu(1)],
        state_names=>$state_names
    );
    $mod->bind(data_shapes=>[['data', [5, 10]]], for_training=>0);
    $mod->init_params();
    my $batch = mx->io->DataBatch(data=>[mx->nd->zeros([5, 10])], label=>[]);

    $mod->set_states(value=>1);
    $mod->forward($batch);
    my $out = $mod->get_outputs(0);
    my $out1 = $mod->get_outputs(1);

    $mod->set_states(states=>$out);
    $mod->forward($batch);
    my $out2 = $mod->get_outputs(1);

    for(zip($out1, $out2)) {
        my ($x1, $x2) = @$_;
        ok(not almost_equal($x1->aspdl, $x2->aspdl, 1e-3));
    }
}

sub test_module_dtype
{
    my $dtype = 'float16';
    my $dshape = [3, 8, 7];

    my $sym = mx->sym->Variable('data');
    $sym    = mx->sym->Activation(data=>$sym, act_type=>'relu', __layout__=>'TNC');

    my $mod = mx->mod->Module($sym, data_names=>['data'], context => [mx->cpu(0), mx->cpu(1)]);
    $mod->bind(data_shapes=>[
        mx->io->DataDesc('data', $dshape, dtype => $dtype, layout=>'TNC')
    ]);
    $mod->init_params();
    $mod->forward(
        mx->io->DataBatch(
            data=>[mx->nd->ones($dshape, dtype=>$dtype)]
        )
    );
    $mod->backward([mx->nd->ones($dshape, dtype=>$dtype)]);

    for my $x (@{ $mod->get_outputs() })
    {
        is($x->dtype, $dtype);
    }
}

sub test_module_input_grads
{
    my $a = mx->sym->Variable('a', __layout__=>'NC');
    my $b = mx->sym->Variable('b', __layout__=>'NC');
    my $c = mx->sym->Variable('c', __layout__=>'NC');

    $c = $a + 2 * $b + 3 * $c;
    my $net = mx->mod->Module(
        $c, data_names=>['b', 'c', 'a'],
        context=>[mx->cpu(0), mx->cpu(1)]
    );
    $net->bind(
        data_shapes      => [['b', [5, 5]], ['c', [5, 5]], ['a', [5, 5]]],
        inputs_need_grad => 1
    );
    $net->init_params();

    $net->forward(
        mx->io->DataBatch(data => [
            mx->nd->ones([5, 5]),
            mx->nd->ones([5, 5]),
            mx->nd->ones([5, 5])
        ])
    );
    $net->backward([mx->nd->ones([5, 5])]);
    my $input_grads = $net->get_input_grads();
    my $b_grad = $input_grads->[0]->aspdl;
    my $c_grad = $input_grads->[1]->aspdl;
    my $a_grad = $input_grads->[2]->aspdl;
    ok(($a_grad == 1)->all);
    ok(($b_grad == 2)->all);
    ok(($c_grad == 3)->all);
}

sub test_forward_acceptable_input
{
    my $data = mx->sym->Variable('data');
    my $out = $data * 2;
    my $mod = mx->mod->Module(symbol => $out);
    $mod->bind(data_shapes => [['data', [1, 10]]]);
    $mod->init_params();
    is_deeply($mod->predict(mx->nd->ones([1, 10]))->shape, [1, 10]);
    is_deeply($mod->predict(mx->nd->ones([1, 10])->aspdl)->shape, [1, 10]);
}

test_module_input_grads();
test_module_dtype();
test_module_layout();
test_module_states();
test_save_load();
test_forward_acceptable_input();
