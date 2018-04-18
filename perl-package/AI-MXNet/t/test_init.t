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
use Test::More tests => 7;
use AI::MXNet qw(mx);

sub test_default_init
{
    my $data = mx->sym->Variable('data');
    my $sym  = mx->sym->LeakyReLU(data => $data, act_type => 'prelu');
    my $mod  = mx->mod->Module($sym);
    $mod->bind(data_shapes=>[['data', [10,10]]]);
    $mod->init_params;
    ok((((values %{ ($mod->get_params)[0] }))[0]->aspdl == 0.25)->all);
}

sub test_variable_init
{
    my $data  = mx->sym->Variable('data');
    my $gamma = mx->sym->Variable('gamma', init => mx->init->One());
    my $sym   = mx->sym->LeakyReLU(data => $data, gamma => $gamma, act_type => 'prelu');
    my $mod   = mx->mod->Module($sym);
    $mod->bind(data_shapes=>[['data', [10,10]]]);
    $mod->init_params();
    ok(((values %{ ($mod->get_params)[0] })[0]->aspdl == 1)->all);
}

sub test_aux_init
{
    my $data = mx->sym->Variable('data');
    my $sym  = mx->sym->BatchNorm(data => $data, name => 'bn');
    my $mod  = mx->mod->Module($sym);
    $mod->bind(data_shapes=>[['data', [10, 10, 3, 3]]]);
    $mod->init_params();
    ok((($mod->get_params)[1]->{bn_moving_var}->aspdl == 1)->all);
    ok((($mod->get_params)[1]->{bn_moving_mean}->aspdl == 0)->all);
}

$ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE} = 0;
sub test_rsp_const_init
{
    my $check_rsp_const_init = sub { my ($init, $val) = @_;
        my $shape = [10, 10];
        my $x = mx->symbol->Variable("data", stype=>'csr');
        my $weight = mx->symbol->Variable("weight", shape=>[$shape->[1], 2],
                                    init=>$init, stype=>'row_sparse');
        my $dot = mx->symbol->sparse->dot($x, $weight);
        my $mod = mx->mod->Module($dot);
        $mod->bind(data_shapes=>[['data', $shape]]);
        $mod->init_params;
        ok(((values %{ ($mod->get_params)[0] })[0]->aspdl == $val)->all);
    };
    $check_rsp_const_init->(mx->initializer->Constant(value=>2), 2);
    $check_rsp_const_init->(mx->initializer->Zero(), 0);
    $check_rsp_const_init->(mx->initializer->One(), 1);
}

test_rsp_const_init();
test_default_init();
test_variable_init();
test_aux_init();
