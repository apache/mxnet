use strict;
use warnings;
use Test::More tests => 1;
use AI::MXNet qw(mx);
use Data::Dumper;
sub test_inits
{
    my $arr = mx->nd->zeros([5, 5, 4, 4]);
    my $MSRAPrelu = mx->init->MSRAPrelu;
    &{$MSRAPrelu}("upsampling", $arr);
    print $arr->aspdl;
    &{$MSRAPrelu}("stn_loc_weight", $arr);
    print $arr->aspdl;
    $arr = mx->nd->zeros([6]);
    &{$MSRAPrelu}("stn_loc_bias", $arr);
    print $arr->aspdl;
    $arr = mx->nd->zeros([5,5]);
    &{$MSRAPrelu}("weight", $arr);
    print $arr->aspdl;
    my $Xavier = mx->init->Xavier;
    &{$Xavier}("weight", $arr);
    print $arr->aspdl;
    my $Uniform = mx->init->Uniform;
    &{$Uniform}("weight", $arr);
    print $arr->aspdl;
    my $Normal = mx->init->Normal;
    &{$Normal}("weight", $arr);
    print $arr->aspdl;
    $arr = mx->nd->zeros([4,5,6]);
    my $Orthogonal = mx->init->Orthogonal;
    &{$Orthogonal}("weight", $arr);
    print $arr->aspdl;
}

test_inits();

ok(1);