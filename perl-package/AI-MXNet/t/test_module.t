use strict;
use warnings;
use Test::More tests => 247;
use AI::MXNet qw(mx);
use AI::MXNet::Base;
use AI::MXNet::TestUtils qw(almost_equal enumerate same_array);
use Data::Dumper;

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

sub test_monitor
{
    mx->random->seed(11);
    my $data = mx->nd->array([[0.05, .10]]);
    my $label = mx->nd->array([[.01, 0.99]]);
    my $train_data = mx->io->NDArrayIter($data, label => $label, batch_size=>1);

    # symbols
    my $x = mx->symbol->Variable('data');
    $x = mx->symbol->FullyConnected(name=>'fc_0', data=>$x, num_hidden=>2);
    $x = mx->symbol->Activation(name=>"act_0", data=>$x, act_type=>'sigmoid');
    $x = mx->symbol->FullyConnected(name=>'fc_1', data=>$x, num_hidden=>2);
    $x = mx->symbol->Activation(name=>"act_1", data=>$x, act_type=>'sigmoid');
    $x = mx->symbol->LinearRegressionOutput(data=>$x, name=>'softmax', grad_scale=>2);

    # create monitor
    my $mean_abs = sub { my ($x) = @_;
        return $x->abs->sum/$x->size;
    };
    my $mon = mx->mon->Monitor(1, stat_func=>$mean_abs, pattern=>'.*', sort=>1);

    # create module
    my $mod = mx->mod->Module($x, context=>[mx->cpu()]);
    $mod->bind(data_shapes=>$train_data->provide_data, label_shapes=>$train_data->provide_label,
                    for_training=>1);
    $mod->install_monitor($mon);
    my $arg_params = {fc_0_weight => mx->nd->array([[.15, .20], [.25, .30]]),
                  fc_0_bias  => mx->nd->array([.35, .35]),
                  fc_1_weight => mx->nd->array([[.40, .45], [.50, .55]]),
                  fc_1_bias  => mx->nd->array([.60, .60])};
    $mod->init_params(arg_params=>$arg_params);

    my $data_batch = <$train_data>;
    $mon->tic();
    $mod->forward_backward($data_batch);
    my $res = $mon->toc();
    my $keys = ['act_0', 'act_1', 'data', 'fc_0', 'fc_1', 'softmax'];
    my $mon_result_counts = [0, 0, 0, 0, 0, 0];
    ok(@$res == 21);
    for my $r (@$res)
    {
        my ($n, $k, $v) = @$r;
        enumerate(sub {
            my ($idx, $key) = @_;
            if($k =~ /^$key/)
            {
                $mon_result_counts->[$idx] += 1;
                return;
            }
        }, $keys);
    }
    is_deeply($mon_result_counts, [2, 2, 1, 6, 6, 4]);
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

sub test_executor_group
{
    my $get_rnn_sym = sub { my ($num_layers, $num_words, $num_hidden, $num_embed, $seq_len) = @_;
        my $stack = mx->rnn->SequentialRNNCell();
        for my $i (0..$num_layers-1)
        {
            $stack->add(mx->rnn->LSTMCell(num_hidden=>$num_hidden, prefix=>"lstm_l${i}_"));
        }
        my $data = mx->sym->Variable('data');
        my $label = mx->sym->Variable('softmax_label');
        my $embed = mx->sym->Embedding(data=>$data, input_dim=>$num_words,
                                 output_dim=>$num_embed, name=>'embed');

        $stack->reset();
        my ($outputs, $states) = $stack->unroll($seq_len, inputs=>$embed, merge_outputs=>1);

        my $pred = mx->sym->Reshape($outputs, shape=>[-1, $num_hidden]);
        $pred = mx->sym->FullyConnected(data=>$pred, num_hidden=>$num_words, name=>'pred');

        $label = mx->sym->Reshape($label, shape=>[-1]);
        $pred = mx->sym->SoftmaxOutput(data=>$pred, label=>$label, name=>'softmax');
        return $pred;
    };

    my $test_shared_exec_group = sub { my ($exec_grp_shared, $exec_grp_created, $shared_arg_names, $extra_args) = @_;
        # Test shared data arrays
        for my $i (0..@{ $exec_grp_shared->execs }-1)
        {
            # test same shared_data_arrays for two exec groups
            my $shared_data_array1 = $exec_grp_shared->shared_data_arrays->[$i];
            my $shared_data_array2 = $exec_grp_created->shared_data_arrays->[$i];
            if(defined $extra_args)
            {
                ok(keys(%$shared_data_array1) == @$extra_args);
            }
            ok(keys(%$shared_data_array1) == keys(%$shared_data_array2));
            while(my ($k, $v) = each %{ $shared_data_array1 })
            {
                if(defined $extra_args)
                {
                    ok(grep { $_ eq $k } @$extra_args);
                }
                ok(exists $shared_data_array2->{$k});
                ok(same_array($v, $shared_data_array2->{$k}));
            }
            # Test shared argument arrays and gradient arrays
            my $exec_shared  = $exec_grp_shared->execs->[$i];
            my $exec_created = $exec_grp_created->execs->[$i];
            if(defined $shared_arg_names)
            {
                # test shared arguments
                for my $arg_name (@$shared_arg_names)
                {
                    ok(exists $exec_created->arg_dict->{$arg_name});
                    ok(same_array($exec_shared->arg_dict->{$arg_name}, $exec_created->arg_dict->{$arg_name}));
                }
                # test shared argument gradients
                for my $arg_name (@$shared_arg_names)
                {
                    ok(exists $exec_created->grad_dict->{$arg_name});
                    ok(same_array($exec_shared->grad_dict->{$arg_name}, $exec_created->grad_dict->{$arg_name}));
                }
            }
            my $grad_req = $exec_grp_shared->grad_req;
            while(my ($arg_name, $grad) = each %{ $grad_req })
            {
                ok($grad eq $exec_grp_created->grad_req->{$arg_name});
            }
        }
    };
    my $contexts = [mx->cpu(0), mx->cpu(1)];
    my $workload = [(1) x scalar(@$contexts)];
    my $batch_size = 32;
    my $max_bucket_size = 80;
    my $num_words = 1000;
    my $num_hidden = 100;
    my $num_embed = 200;
    my $data_shapes = [['data', [$batch_size, $max_bucket_size]]];
    my $label_shapes = [['softmax_label', [$batch_size, $max_bucket_size]]];

    # generate an rnn sym with #layers=5
    my $sym = $get_rnn_sym->(3, $num_words, $num_hidden,
                      $num_embed, $max_bucket_size);
    my $arg_names1 = $sym->list_arguments();
    my $input_names = ['data', 'softmax_label'];
    my $shared_arg_names = [grep { !/^(?:data|softmax_label)$/ } @$arg_names1];
    my $exec_group1 = AI::MXNet::DataParallelExecutorGroup->new(
        symbol=>$sym, contexts=>$contexts,
        workload=>$workload, data_shapes=>$data_shapes,
        label_shapes=>$label_shapes, param_names=>$shared_arg_names,
        for_training=>1, inputs_need_grad=>0
    );
    # shared_data_arrays should only have input "data" and "softmax_label" arrays
    for my $i (0..@{$contexts}-1)
    {
        ok(keys(%{$exec_group1->shared_data_arrays->[$i]}) == @$input_names);
        for my $name (@$input_names)
        {
            ok(exists $exec_group1->shared_data_arrays->[$i]->{$name});
        }
    }
    # generate an rnn sym with #layers=5
    $sym = $get_rnn_sym->(5, $num_words, $num_hidden,
                      $num_embed, $max_bucket_size);
    my $arg_names2 = $sym->list_arguments();
    my $exec_group2 = AI::MXNet::DataParallelExecutorGroup->new(symbol=>$sym, contexts=>$contexts,
                                            workload=>$workload, data_shapes=>$data_shapes,
                                            label_shapes=>$label_shapes, param_names=>$shared_arg_names,
                                            for_training=>1, inputs_need_grad=>0,
                                            shared_group=>$exec_group1);
    my %shared_arg_names = map { $_ => 1 } @$shared_arg_names;
    my $extra_args = [grep { not exists $shared_arg_names{$_} } @$arg_names2];
    $test_shared_exec_group->(
        $exec_group1, $exec_group2,
        $shared_arg_names, $extra_args
    );
}

test_module_input_grads();
test_module_dtype();
test_monitor();
test_module_switch_bucket();
test_module_layout();
test_module_states();
test_module_reshape();
test_save_load();
test_executor_group();
