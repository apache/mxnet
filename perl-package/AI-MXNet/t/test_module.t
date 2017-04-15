use strict;
use warnings;
use Test::More tests => 17;
use AI::MXNet qw(mx);
use AI::MXNet::Base;
use AI::MXNet::TestUtils qw(almost_equal);

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

    zip(sub {
        my ($x1, $x2) = @_;
        ok(not almost_equal($x1->aspdl, $x2->aspdl, 1e-3));
    }, $out1, $out2);
}

sub test_module_switch_bucket
{
    my $vocab_dim  = 5000;
    my $num_hidden = 100;
    my $num_embedding = 100;
    my $num_layer = 2;
    my $default_key = 10;
    my $test_key = 5;
    my $batch_size = 32;
    my $contexts = [mx->cpu(0)];
    my $initializer = mx->init->Xavier(factor_type=>"in", magnitude=>2.34);

    #generate symbols for an LSTM network
    my $gen_sym = sub {
        my $seq_len = shift;
        my $data  = mx->sym->Variable('data');
        my $label = mx->sym->Variable('softmax_label');
        my $embed = mx->sym->Embedding(data=>$data, input_dim=>$vocab_dim,
                                 output_dim=>$num_embedding, name=>'embed');
        my $stack = mx->rnn->SequentialRNNCell();
        for my $i (0..$num_layer-1)
        {
            $stack->add(mx->rnn->LSTMCell(num_hidden=>$num_hidden, prefix=>"lstm_l${i}_"));
        }
        my ($outputs, $states) = $stack->unroll($seq_len, inputs=>$embed, merge_outputs=>1);

        my $pred = mx->sym->Reshape($outputs, shape=>[-1, $num_hidden]);
        $pred = mx->sym->FullyConnected(data=>$pred, num_hidden=>$vocab_dim, name=>'pred');

        $label = mx->sym->Reshape($label, shape=>[-1]);
        $pred = mx->sym->SoftmaxOutput(data=>$pred, label=>$label, name=>'softmax');

        return ($pred, ['data'], ['softmax_label']);
    };
    my $create_bucketing_module = sub { my $key = shift;
        my $model = mx->mod->BucketingModule(
            sym_gen             => $gen_sym,
            default_bucket_key  => $key,
            context             => $contexts
        );
        $model->bind(data_shapes=>[['data', [$batch_size, $key]]],
                    label_shapes=>[['softmax_label', [$batch_size, $key]]]
        );
        $model->init_params(initializer=>$initializer);
        return $model;
    };
    #initialize the bucketing module with the default bucket key
    my $bucketing_model = $create_bucketing_module->($default_key);
    #switch to test_key
    $bucketing_model->switch_bucket(
        bucket_key   => $test_key,
        data_shapes  => [['data', [$batch_size, $test_key]]],
        label_shapes => [['softmax_label', [$batch_size, $test_key]]]
    );

    delete $bucketing_model->_buckets->{$test_key};

    $bucketing_model->switch_bucket(
        bucket_key   => $test_key,
        data_shapes  => [['data', [$batch_size, $test_key]]],
        label_shapes => [['softmax_label', [$batch_size, $test_key]]]
    );
}

test_module_switch_bucket();
test_module_layout();
test_module_states();
test_module_reshape();
test_save_load();
