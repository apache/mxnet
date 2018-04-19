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
use Test::More tests => 30;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::TestUtils qw(almost_equal dies_ok);
use Scalar::Util qw(refaddr);
use AI::MXNet::Base;

sub test_multi_trainer
{
    my $x = gluon->Parameter('x', shape=>[10], stype=>'row_sparse');
    $x->initialize();
    # test set trainer
    my $trainer0 = gluon->Trainer([$x], 'sgd');
    ok(refaddr($x->_trainer) == refaddr($trainer0));
    # test unset trainer
    $x->_set_trainer(undef);
    ok(not defined $x->_trainer);
    $x->_set_trainer($trainer0);
    # multiple trainers for a sparse Parameter are not allowed
    dies_ok(sub { gluon->Trainer([$x], 'sgd') });
}

sub test_trainer
{
    my $dict_equ = sub { my ($a, $b) = @_;
        is_deeply({ map { $_ => 1 } keys %$a }, { map { $_ => 1 } keys %$b });
        for my $k (keys %$a)
        {
            ok(($a->{$k}->aspdl == $b->{$k}->aspdl)->all);
        }
    };
    my $x = gluon->Parameter('x', shape=>[10]);
    $x->initialize(ctx=>[mx->cpu(0), mx->cpu(1)], init=>'zeros');
    my $trainer = gluon->Trainer([$x], 'sgd', {'learning_rate'=> 1.0, 'momentum'=> 0.5});
    my $y;
    mx->autograd->record(sub {
        for my $w (@{ $x->list_data() })
        {
            $y = $w + 1;
            $y->backward();
        }
    });
    $trainer->step(1);

    ok(($x->data(mx->cpu(1))->aspdl == -2)->all);

    $x->lr_mult(0.5);

    mx->autograd->record(sub {
        for my $w (@{ $x->list_data() })
        {
            $y = $w + 1;
            $y->backward();
        }
    });
    $trainer->step(1);

    ok(($x->data(mx->cpu(1))->aspdl == -4)->all);

    $trainer->save_states('test_trainer.states');
    my $states;
    if($trainer->update_on_kvstore)
    {
        $states = { %{ $trainer->kvstore->_updater->states } };
    }
    else
    {
        $states = { %{ $trainer->_updaters->[0]->states } };
    }
    $trainer->load_states('test_trainer.states');
    if($trainer->update_on_kvstore)
    {
        $dict_equ->($trainer->kvstore->_updater->states, $states);
        ok($trainer->_optimizer eq $trainer->kvstore->_updater->optimizer);
    }
    else
    {
        for my $updater (@{ $trainer->_updaters })
        {
            $dict_equ->($updater->states, $states);
        }
        ok($trainer->_optimizer eq $trainer->_updaters->[0]->optimizer);
    }

    dies_ok(sub { $trainer->update(1 ) });
    dies_ok(sub { $trainer->allreduce_grads() });

    $x = gluon->Parameter('x', shape=>[10]);
    $x->initialize(ctx=>[mx->cpu(0), mx->cpu(1)], init=>'zeros');
    my $trainer2 = gluon->Trainer([$x], 'sgd', {learning_rate => 1.0, momentum => 0.5},
                             update_on_kvstore=>0);
    mx->autograd->record(sub {
        for(enumerate($x->list_data))
        {
            my ($i, $w) = @$_;
            my $y = $i*$w;
            $y->backward;
        }
    });
    ok(($x->grad(mx->cpu(0))->aspdl != $x->grad(mx->cpu(1))->aspdl)->all);
    $trainer2->allreduce_grads;
    ok(($x->grad(mx->cpu(0))->aspdl == $x->grad(mx->cpu(1))->aspdl)->all);
    $trainer2->update(1);
    ok(($x->data(mx->cpu(1))->aspdl == -1)->all);

}

test_trainer();

sub test_trainer_save_load
{
    my $x = gluon->Parameter('x', shape=>[10], lr_mult=>1.0);
    $x->initialize(ctx=>[mx->cpu(0), mx->cpu(1)], init=>'zeros');
    my $trainer = gluon->Trainer([$x], 'sgd', {learning_rate => 0.1});
    mx->autograd->record(sub {
        for my $w (@{ $x->list_data })
        {
            my $y = $w + 1;
            $y->backward();
        }
    });
    $trainer->step(1);
    ok($trainer->kvstore->_updater->optimizer->_get_lr(0) == 0.1);
    $trainer->save_states('test_trainer_save_load.states');
    $trainer->load_states('test_trainer_save_load.states');
    $x->lr_mult(2.0);
    # check if parameter dict is correctly associated with optimizer after load_state
    ok($trainer->kvstore->_updater->optimizer->_get_lr(0) == 0.2);
}

test_trainer_save_load();

sub test_trainer_multi_layer_init
{
    local($ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE}) = 0;
    package Net {
        use AI::MXNet::Gluon::Mouse;
        extends 'AI::MXNet::Gluon::Block';
        use AI::MXNet::Function::Parameters;
        sub BUILD {
            my $self = shift;
            $self->name_scope(sub {
                # sparse param
                $self->embed_weight($self->params->get('embed_weight', stype=>'row_sparse',
                                                    shape=>[4,3], grad_stype=>'row_sparse'));
                # dense param from a hybrid block
                $self->dense0(nn->Dense(2));
            });
        }
        method forward($x)
        {
            my $embed_weight = $self->embed_weight->row_sparse_data($x);
            my $embed = mx->nd->Embedding(data=>$x, weight=>$embed_weight,
                                    input_dim=>4, output_dim=>3, sparse_grad=>1);
            return $self->dense0->($embed);
        }
    };
    my $check_init = sub { my ($ctxes) = @_;
        my $net = Net->new(prefix=>'net_');
        $net->initialize(mx->init->One(), ctx=>$ctxes);
        my $trainer = gluon->Trainer($net->collect_params(), 'sgd', {learning_rate => 1});
        my $data = mx->nd->array([[0,2], [1,2]]);
        my $xs = gluon->utils->split_and_load($data, ctx_list => $ctxes);
        my @ys;
        mx->autograd->record(sub {
            for my $x (@{ $xs })
            {
                my $y = $net->($x);
                push @ys, $y;
            }
        });
        for my $y (@ys)
        {
            $y->backward;
        }
        $trainer->step(1);
        # all parameters should be initialized
        ok(not @{ $trainer->_params_to_init });
        my $all_rows = mx->nd->arange(start => 0, stop => 4, ctx=>mx->cpu(1));
        # check the updated weights
        my $weight = $net->embed_weight->row_sparse_data($all_rows)->aspdl;
        ok(($weight->at(0) == -1)->all);
        ok(($weight->at(1) == -1)->all);
        ok(($weight->at(2) == -3)->all);
        ok(($weight->at(3) ==  1)->all);
    };
    $check_init->([mx->cpu(1), mx->cpu(2)]);
    $check_init->([mx->cpu(1)]);
}

test_trainer_multi_layer_init();

sub test_trainer_reset_kv
{
    my $check_trainer_reset_kv = sub { my ($kv) = @_;
        my $params = gluon->ParameterDict();
        my $x = $params->get('x', shape=>[10], lr_mult=>1.0);
        $params->initialize(ctx=>[mx->cpu(0), mx->cpu(1)], init=>'zeros');
        my $trainer = gluon->Trainer($params, 'sgd', {learning_rate => 0.1}, kvstore=>$kv);
        $params->save('test_trainer_reset_kv.params');
        mx->autograd->record(sub {
            for my $w (@{ $x->list_data })
            {
                my $y = $w + 1;
                $y->backward;
            }
        });
        $trainer->step(1);
        is($trainer->kvstore->type, $kv);
        # load would reset kvstore
        $params->load('test_trainer_reset_kv.params', ctx => [mx->cpu(0), mx->cpu(1)]);
        ok(not defined $trainer->kvstore);
        ok (defined $trainer->_kv_initialized and not $trainer->_kv_initialized);
        mx->autograd->record(sub {
            for my $w (@{ $x->list_data })
            {
                my $y = $w + 1;
                $y->backward;
            }
        });
        $trainer->step(1);
        # the updated parameter should be based on the loaded checkpoint
        ok(($x->data(mx->cpu()) == -0.2)->aspdl->all);
    };
    my @kvs = ('local', 'device');
    for my $kv (@kvs)
    {
        $check_trainer_reset_kv->($kv);
    }
}

test_trainer_reset_kv();

