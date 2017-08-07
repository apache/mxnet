use strict;
use warnings;
use Test::More tests => 4;
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
    ok((((values %{ ($mod->get_params)[0] }))[0]->aspdl == 1)->all);
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

test_default_init();
test_variable_init();
test_aux_init();
