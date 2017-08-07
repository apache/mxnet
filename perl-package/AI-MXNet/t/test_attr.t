use strict;
use warnings;
use Test::More tests => 14;
use AI::MXNet qw(mx);
use Storable;

sub contains
{
    my ($x, $y) = @_;
    while(my ($k, $v) = each %$x)
    {
        return 0 unless exists $y->{$k};
        if(ref $y->{$k} and ref $y->{$k} eq 'HASH')
        {
            return 0 unless (ref $v and ref $v eq 'HASH');
            return 0 unless contains($v, $y->{$k});
        }
        elsif($y->{$k} ne $v)
        {
            return 0;
        }
    }
    return 1;
}

sub test_attr_basic
{
    my ($data, $gdata);
    {
        local($mx::AttrScope) = mx->AttrScope(group=>'4', data=>'great');
        $data = mx->symbol->Variable(
            'data',
            attr => {
                qw/ dtype data
                    group 1
                    force_mirroring 1/
            },
            lr_mult => 1);
        $gdata = mx->symbol->Variable('data2');
    }
    ok($gdata->attr('group') == 4);
    ok($data->attr('group') == 1);
    ok($data->attr('lr_mult') == 1);
    ok($data->attr('__lr_mult__') == 1);
    ok($data->attr('force_mirroring') == 1);
    ok($data->attr('__force_mirroring__') == 1);
    my $data2 = Storable::thaw(Storable::freeze($data));
    ok($data->attr('dtype') eq $data2->attr('dtype'));
}

sub test_operator
{
    my $data = mx->symbol->Variable('data');
    my ($fc1, $fc2);
    {
        local($mx::AttrScope) = mx->AttrScope(__group__=>'4', __data__=>'great');
        $fc1 = mx->symbol->Activation($data, act_type=>'relu');
        {
            local($mx::AttrScope) = mx->AttrScope(__init_bias__ => 0, 
                __group__=>'4', __data__=>'great');
            $fc2 = mx->symbol->FullyConnected($fc1, num_hidden=>10, name=>'fc2');
        }
    }
    ok($fc1->attr('__data__') eq 'great');
    ok($fc2->attr('__data__') eq 'great');
    ok($fc2->attr('__init_bias__') == 0);
    my $fc2copy = Storable::thaw(Storable::freeze($fc2));
    ok($fc2copy->tojson() eq $fc2->tojson());
    ok($fc2->get_internals()->slice('fc2_weight'));
}

sub test_list_attr
{
    my $data = mx->sym->Variable('data', attr=>{'mood', 'angry'});
    my $op = mx->sym->Convolution(
        data=>$data, name=>'conv', kernel=>[1, 1],
        num_filter=>1, attr => {'__mood__'=> 'so so', 'wd_mult'=> 'x'}
    );
    ok(contains({'__mood__'=> 'so so', 'wd_mult'=> 'x', '__wd_mult__'=> 'x'}, $op->list_attr()));
}

sub test_attr_dict
{
    my $data = mx->sym->Variable('data', attr=>{'mood'=> 'angry'});
    my $op = mx->sym->Convolution(
        data=>$data, name=>'conv', kernel=>[1, 1],
        num_filter=>1, attr=>{'__mood__'=> 'so so'}, lr_mult=>1
    );
    ok(
        contains(
            {
                'data'=> {'mood'=> 'angry'},
                'conv_weight'=> {'__mood__'=> 'so so'},
                'conv'=> {
                    'kernel'=> '(1, 1)', '__mood__'=> 'so so', 
                    'num_filter'=> '1', 'lr_mult'=> '1', '__lr_mult__'=> '1'
                },
                'conv_bias'=> {'__mood__'=> 'so so'}
            },
            $op->attr_dict()
        )
    );
}

test_attr_basic();
test_operator();
test_list_attr();
test_attr_dict();
