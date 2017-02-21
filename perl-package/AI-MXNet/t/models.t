use strict;
use warnings;
use Test::More tests => 2;
use AI::MXNet qw(mx);
use Data::Dumper;
sub mlp2
{
    my $data = mx->symbol->Variable('data');
    my $out = mx->symbol->FullyConnected($data, name => 'fc1', num_hidden => 15);
    $out = mx->symbol->Activation($out, act_type => 'relu');
    $out = mx->symbol->FullyConnected($out, name => 'fc2', num_hidden => 10);
    $out->simple_bind(type_dict => { data => "float32" }, shapes => { data => [1,10] });
}

sub conv
{
    my $data = mx->symbol->Variable('data');
    my $conv1 = mx->symbol->Convolution($data, name => 'conv1', num_filter => 32, kernel => [3,3], stride =>[2,2]);
    my $bn1 = mx->symbol->BatchNorm($conv1, name => "bn1");
    my $act1 = mx->symbol->Activation($bn1, name => 'relu1', act_type => "relu");
    my $mp1 = mx->symbol->Pooling($act1, name => 'mp1', kernel => [2,2], stride=>[2,2], pool_type=>'max');

    my $conv2 = mx->symbol->Convolution($mp1, name=>'conv2', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn2 = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2 = mx->symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2 = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $fl = mx->symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc2 = mx->symbol->FullyConnected(data => $fl, name=>'fc2', num_hidden=>10);
    my $softmax = mx->symbol->SoftmaxOutput({ data => $fc2, name => 'sm' });
    $softmax->simple_bind(type_dict => { data => "float32" }, shapes => { data => [28,1,28,28] });
}

is_deeply(mlp2()->forward(1, data => [[0..9]])->[0]->aspdl->unpdl, [
          [
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0'
          ]
        ]);

is_deeply(conv()->forward()->[0]->aspdl->unpdl, [ map { [ map { 0.100000001490116 } 0..9 ] } 0..27 ]);
