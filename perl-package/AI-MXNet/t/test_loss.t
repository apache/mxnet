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
use Test::More tests => 24;
use AI::MXNet 'mx';
use AI::MXNet::Gluon 'gluon';
use AI::MXNet::TestUtils 'almost_equal';
use Hash::Ordered;

sub test_loss_ndarray
{
    my $output     = mx->nd->array([1, 2, 3, 4]);
    my $label      = mx->nd->array([1, 3, 5, 7]);
    my $weighting  = mx->nd->array([0.5, 1, 0.5, 1]);

    my $loss = gluon->loss->L1Loss();
    ok(mx->nd->sum($loss->($output, $label))->asscalar() == 6);
    $loss = gluon->loss->L1Loss(weight=>0.5);
    ok(mx->nd->sum($loss->($output, $label))->asscalar() == 3);
    $loss = gluon->loss->L1Loss();
    ok(mx->nd->sum($loss->($output, $label, $weighting))->asscalar() == 5);

    $loss = gluon->loss->L2Loss();
    ok(mx->nd->sum($loss->($output, $label))->asscalar() == 7);
    $loss = gluon->loss->L2Loss(weight=>0.25);
    ok(mx->nd->sum($loss->($output, $label))->asscalar() == 1.75);
    $loss = gluon->loss->L2Loss();
    ok(mx->nd->sum($loss->($output, $label, $weighting))->asscalar() == 6);

    $output    = mx->nd->array([[0, 2], [1, 4]]);
    $label     = mx->nd->array([0, 1]);
    $weighting = mx->nd->array([[0.5], [1.0]]);

    $loss = gluon->loss->SoftmaxCrossEntropyLoss();
    my $L = $loss->($output, $label)->aspdl();
    ok(almost_equal($L, mx->nd->array([ 2.12692809,  0.04858733])->aspdl));

    $L = $loss->($output, $label, $weighting)->aspdl();
    ok(almost_equal($L, mx->nd->array([ 1.06346405,  0.04858733])->aspdl));
}

test_loss_ndarray();

sub get_net
{
    my ($num_hidden, $flatten) = @_;
    $flatten //= 1;
    my $data = mx->symbol->Variable('data');
    my $fc1 = mx->symbol->FullyConnected($data, name=>'fc1', num_hidden=>128, flatten=>$flatten);
    my $act1 = mx->symbol->Activation($fc1, name=>'relu1', act_type=>"relu");
    my $fc2 = mx->symbol->FullyConnected($act1, name => 'fc2', num_hidden => 64, flatten=>$flatten);
    my $act2 = mx->symbol->Activation($fc2, name=>'relu2', act_type=>"relu");
    my $fc3 = mx->symbol->FullyConnected($act2, name=>'fc3', num_hidden=>$num_hidden, flatten=>$flatten);
    return $fc3;
}

sub test_ce_loss
{
    my $nclass = 10;
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, $nclass]);
    my $label = mx->nd->array([qw/3 6 5 4 8 9 1 7 9 6 8 0 5 0 9 6 2 0 5 2/], dtype=>'int32');
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label');
    my $output = get_net($nclass);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->SoftmaxCrossEntropyLoss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    local($AI::MXNet::Logging::silent) = 1;
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam');
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.1);
}

test_ce_loss();

sub test_bce_loss
{
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, 20]);
    my $label = mx->nd->array([qw/1 1 0 1 0 0 0 1 1 1 1 1 0 0 1 0 0 0 0 0/], dtype=>'float32');
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label');
    my $output = get_net(1);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->SigmoidBinaryCrossEntropyLoss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    local($AI::MXNet::Logging::silent) = 1;
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam',
            initializer=>mx->init->Xavier(magnitude=>2));
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.01);
}

test_bce_loss();

sub test_bce_equal_ce2
{
    my $N = 100;
    my $loss1 = gluon->loss->SigmoidBCELoss(from_sigmoid=>1);
    my $loss2 = gluon->loss->SoftmaxCELoss(from_logits=>1);
    my $out1 = mx->random->uniform(0, 1, shape=>[$N, 1]);
    my $out2 = mx->nd->log(mx->nd->concat(1-$out1, $out1, dim=>1) + 1e-8);
    my $label = mx->nd->round(mx->random->uniform(0, 1, shape=>[$N, 1]));
    ok(almost_equal($loss1->($out1, $label)->aspdl, $loss2->($out2, $label)->aspdl));
}

test_bce_equal_ce2();

sub test_kl_loss
{
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, 10]);
    my $label = mx->nd->softmax(mx->random->uniform(0, 1, shape=>[$N, 2]));
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label');
    my $output = mx->sym->log_softmax(get_net(2));
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->KLDivLoss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam');
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.05);
}

test_kl_loss();

sub test_l2_loss
{
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, 10]);
    my $label = mx->nd->softmax(mx->random->uniform(-1, 1, shape=>[$N, 1]));
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label', shuffle=>1);
    my $output = get_net(1);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->L2Loss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam');
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.1);
}

test_l2_loss();

sub test_l1_loss
{
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, 10]);
    my $label = mx->nd->softmax(mx->random->uniform(-1, 1, shape=>[$N, 1]));
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label', shuffle=>1);
    my $output = get_net(1);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->L1Loss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam');
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.1);
}

test_l1_loss();

sub test_ctc_loss
{
    my $loss = gluon->loss->CTCLoss();
    my $l = $loss->(mx->nd->ones([2,20,4]), mx->nd->array([[1,0,-1,-1],[2,1,1,-1]]));
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));

    $loss = gluon->loss->CTCLoss(layout=>'TNC');
    $l = $loss->(mx->nd->ones([20,2,4]), mx->nd->array([[1,0,-1,-1],[2,1,1,-1]]));
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));

    $loss = gluon->loss->CTCLoss(layout=>'TNC', label_layout=>'TN');
    $l = $loss->(mx->nd->ones([20,2,4]), mx->nd->array([[1,0,-1,-1],[2,1,1,-1]])->T);
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));

    $loss = gluon->loss->CTCLoss();
    $l = $loss->(mx->nd->ones([2,20,4]), mx->nd->array([[2,1,2,2],[3,2,2,2]]), undef, mx->nd->array([2,3]));
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));

    $loss = gluon->loss->CTCLoss();
    $l = $loss->(mx->nd->ones([2,25,4]), mx->nd->array([[2,1,-1,-1],[3,2,2,-1]]), mx->nd->array([20,20]));
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));

    $loss = gluon->loss->CTCLoss();
    $l = $loss->(mx->nd->ones([2,25,4]), mx->nd->array([[2,1,3,3],[3,2,2,3]]), mx->nd->array([20,20]), mx->nd->array([2,3]));
    ok(almost_equal($l->aspdl, mx->nd->array([18.82820702, 16.50581741])->aspdl));
}

test_ctc_loss();

sub test_ctc_loss_train
{
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, 20, 10]);
    my $label = mx->nd->arange(start => 4, repeat=>$N)->reshape([$N, 4]);
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label', shuffle=>1);
    my $output = get_net(5, 0);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->CTCLoss(layout=>'NTC', label_layout=>'NT');
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 1},
            initializer=>mx->init->Xavier(magnitude=>2), eval_metric=>mx->metric->Loss(),
            optimizer=>'adam');
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 20);
}

test_ctc_loss_train();

sub test_sample_weight_loss
{
    my $nclass = 10;
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, $nclass]);
    my $label = mx->nd->array([qw/2 0 8 4 3 4 2 5 5 7 2 3 7 1 2 6 4 2 8 0/], dtype=>'int32');
    my $weight = mx->nd->array([(1)x10,(0)x10]);
    my $data_iter = mx->io->NDArrayIter(
        $data,
        Hash::Ordered->new(label => $label, w => $weight),
        batch_size=>10
    );
    my $output = get_net($nclass);
    my $l = mx->symbol->Variable('label');
    my $w = mx->symbol->Variable('w');
    my $Loss = gluon->loss->SoftmaxCrossEntropyLoss();
    my $loss = $Loss->($output, $l, $w);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label', 'w']);
    $mod->fit($data_iter, num_epoch=>200, optimizer_params=>{learning_rate => 0.01},
            eval_metric=>mx->metric->Loss(), optimizer=>'adam');
    $data_iter = mx->io->NDArrayIter(
        $data->slice([10,$data->len-1]),
        Hash::Ordered->new(label => $label, w => $weight),
        batch_size=>10
    );
    my $score =  $mod->score($data_iter, mx->metric->Loss())->{loss};
    ok($score > 1);
    $data_iter = mx->io->NDArrayIter(
        $data->slice([0,9]),
        Hash::Ordered->new(label => $label, w => $weight),
        batch_size=>10
    );
    $score =  $mod->score($data_iter, mx->metric->Loss())->{loss};
    ok($score < 0.05);
}

test_sample_weight_loss();

sub test_saveload
{
    mx->random->seed(1234);
    my $nclass = 10;
    my $N = 20;
    my $data = mx->random->uniform(-1, 1, shape=>[$N, $nclass]);
    my $label = mx->nd->array([qw/2 0 8 4 3 4 2 5 5 7 2 3 7 1 2 6 4 2 8 0/], dtype=>'int32');
    my $data_iter = mx->io->NDArrayIter($data, $label, batch_size=>10, label_name=>'label');
    my $output = get_net($nclass);
    my $l = mx->symbol->Variable('label');
    my $Loss = gluon->loss->SoftmaxCrossEntropyLoss();
    my $loss = $Loss->($output, $l);
    $loss = mx->sym->make_loss($loss);
    local($AI::MXNet::Logging::silent) = 1;
    my $mod = mx->mod->Module($loss, data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>100, optimizer_params=>{learning_rate => 1},
            eval_metric=>mx->metric->Loss());
    $mod->save_checkpoint('test', 100, 1);
    $mod = mx->mod->Module->load('test', 100, 1,
                             data_names=>['data'], label_names=>['label']);
    $mod->fit($data_iter, num_epoch=>100, optimizer_params=>{learning_rate => 1},
            eval_metric=>mx->metric->Loss()
    );
    ok($mod->score($data_iter, mx->metric->Loss())->{loss} < 0.05);
}

test_saveload();
