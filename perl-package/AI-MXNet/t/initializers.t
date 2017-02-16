use strict;
use warnings;
use Test::More tests => 3;
use AI::MXNet qw(mx);
use Data::Dumper;
sub test_inits
{
    my $arr = mx->nd->zeros([1,1,1,1]);
    my $MSRAPrelu = mx->init->MSRAPrelu;
    &{$MSRAPrelu}("upsampling", $arr);
    is_deeply($arr->aspdl->unpdl, [
          [
            [
              [
                '1'
              ]
            ]
          ]
        ]);

    &{$MSRAPrelu}("stn_loc_weight", $arr);
    is_deeply($arr->aspdl->unpdl,[
          [
            [
              [
                '0'
              ]
            ]
          ]
        ]);

    $arr = mx->nd->zeros([6]);
    &{$MSRAPrelu}("stn_loc_bias", $arr);
    is_deeply($arr->aspdl->unpdl, [
          '1',
          '0',
          '0',
          '0',
          '1',
          '0'
        ]);
}

test_inits();
