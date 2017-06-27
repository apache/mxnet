package AI::MXNet::TestUtils;
use strict;
use warnings;
use PDL;
use Carp;
use Scalar::Util qw(blessed);
use AI::MXNet::Function::Parameters;
use Exporter;
use base qw(Exporter);
@AI::MXNet::TestUtils::EXPORT_OK = qw(same reldiff almost_equal GetMNIST_ubyte
                                      GetCifar10 pdl_maximum pdl_minimum mlp2 conv
                                      check_consistency zip assert enumerate same_array);
use constant default_numerical_threshold => 1e-6;
=head1 NAME

    AI::MXNet::TestUtils - Convenience subs used in tests.

=head2 same

    Test if two pdl arrays are the same

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func same(PDL $a, PDL $b)
{
    return ($a != $b)->sum == 0;
}

=head2 reldiff

    Calculate the relative difference between two input arrays

    Calculated by :math:`\\frac{|a-b|_1}{|a|_1 + |b|_1}`

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func reldiff(PDL $a, PDL $b)
{
    my $diff = sum(abs($a - $b));
    my $norm = sum(abs($a)) + sum(abs($b));
    if($diff == 0)
    {
        return 0;
    }
    my $ret = $diff / $norm;
    return $ret;
}

=head2 almost_equal

    Test if two pdl arrays are almost equal.
=cut

func almost_equal(PDL $a, PDL $b, Maybe[Num] $threshold=)
{
    $threshold //= default_numerical_threshold;
    my $rel = reldiff($a, $b);
    return $rel <= $threshold;
}

func GetMNIST_ubyte()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (
        not -f 'data/train-images-idx3-ubyte'
            or
        not -f 'data/train-labels-idx1-ubyte'
            or
        not -f 'data/t10k-images-idx3-ubyte'
            or
        not -f 'data/t10k-labels-idx1-ubyte'
    )
    {
        `wget http://data.mxnet.io/mxnet/data/mnist.zip -P data`;
        chdir 'data';
        `unzip -u mnist.zip`;
        chdir '..';
    }
}

func GetCifar10()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (not -f 'data/cifar10.zip')
    {
        `wget http://data.mxnet.io/mxnet/data/cifar10.zip -P data`;
        chdir 'data';
        `unzip -u cifar10.zip`;
        chdir '..';
    }
}

func _pdl_compare(PDL $a, PDL|Num $b, Str $criteria)
{
    if(not blessed $b)
    {
        my $tmp = $b;
        $b = $a->copy;
        $b .= $tmp;
    }
    my $mask = {
        'max' => sub { $_[0] < $_[1] },
        'min' => sub { $_[0] > $_[1] },
    }->{$criteria}->($a, $b);
    my $c = $a->copy;
    $c->where($mask) .= $b->where($mask);
    $c;
}

func pdl_maximum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'max');
}

func pdl_minimum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'min');
}

func mlp2()
{
    my $data = AI::MXNet::Symbol->Variable('data');
    my $out  = AI::MXNet::Symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>1000);
    $out     = AI::MXNet::Symbol->Activation(data=>$out, act_type=>'relu');
    $out     = AI::MXNet::Symbol->FullyConnected(data=>$out, name=>'fc2', num_hidden=>10);
    return $out;
}

func conv()
{
    my $data    = AI::MXNet::Symbol->Variable('data');
    my $conv1   = AI::MXNet::Symbol->Convolution(data => $data, name=>'conv1', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn1     = AI::MXNet::Symbol->BatchNorm(data => $conv1, name=>"bn1");
    my $act1    = AI::MXNet::Symbol->Activation(data => $bn1, name=>'relu1', act_type=>"relu");
    my $mp1     = AI::MXNet::Symbol->Pooling(data => $act1, name => 'mp1', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $conv2   = AI::MXNet::Symbol->Convolution(data => $mp1, name=>'conv2', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn2     = AI::MXNet::Symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2    = AI::MXNet::Symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2     = AI::MXNet::Symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $fl      = AI::MXNet::Symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc2     = AI::MXNet::Symbol->FullyConnected(data => $fl, name=>'fc2', num_hidden=>10);
    my $softmax = AI::MXNet::Symbol->SoftmaxOutput(data => $fc2, name => 'sm');
    return $softmax;
}

=head2 check_consistency

    Check symbol gives the same output for different running context

    Parameters
    ----------
    sym : Symbol or list of Symbols
        symbol(s) to run the consistency test
    ctx_list : list
        running context. See example for more detail.
    scale : float, optional
        standard deviation of the inner normal distribution. Used in initialization
    grad_req : str or list of str or dict of str to str
        gradient requirement.
=cut

my %dtypes = (
    float32 => 0,
    float64 => 1,
    float16 => 2,
    uint8   => 3,
    int32   => 4
);

func check_consistency(
    SymbolOrArrayOfSymbols              :$sym,
    ArrayRef                            :$ctx_list,
    Num                                 :$scale=1,
    Str|ArrayRef[Str]|HashRef[Str]      :$grad_req='write',
    Maybe[HashRef[AI::MXNet::NDArray]]  :$arg_params=,
    Maybe[HashRef[AI::MXNet::NDArray]]  :$aux_params=,
    Maybe[HashRef[Num]|Num]             :$tol=,
    Bool                                :$raise_on_err=1,
    Maybe[AI::MXNer::NDArray]           :$ground_truth=
)
{
    $tol //= {
        float16 => 1e-1,
        float32 => 1e-3,
        float64 => 1e-5,
        uint8   => 0,
        int32   => 0
    };
    $tol = {
        float16 => $tol,
        float32 => $tol,
        float64 => $tol,
        uint8   => $tol,
        int32   => $tol
    } unless ref $tol;

    Test::More::ok(@$ctx_list > 1);
    if(blessed $sym)
    {
        $sym = [($sym)x@$ctx_list];
    }
    else
    {
        Test::More::ok(@$sym == @$ctx_list);
    }
    my $output_names = $sym->[0]->list_outputs;
    my $arg_names    = $sym->[0]->list_arguments;
    my @exe_list;
    zip(sub {
        my ($s, $ctx) = @_;
        Test::More::is_deeply($s->list_arguments, $arg_names);
        Test::More::is_deeply($s->list_outputs, $output_names);
        push @exe_list, $s->simple_bind(grad_req=>$grad_req, %$ctx);
    }, $sym, $ctx_list);
    $arg_params //= {};
    $aux_params //= {};
    my %arg_dict = %{ $exe_list[0]->arg_dict };
    while(my ($n, $arr) = each %arg_dict)
    {
        if(not exists $arg_params->{$n})
        {
            $arg_params->{$n} = random(reverse @{ $arr->shape })*$scale;
        }
    }
    my %aux_dict = %{ $exe_list[0]->aux_dict };
    while(my ($n, $arr) = each %aux_dict)
    {
        if(not exists $aux_params->{$n})
        {
            $aux_params->{$n} = 0;
        }
    }
    for my $exe(@exe_list)
    {
        %arg_dict = %{ $exe->arg_dict };
        while(my ($name, $arr) = each %arg_dict)
        {
            $arr .= $arg_params->{$name};
        }
        %aux_dict = %{ $exe->aux_dict };
        while(my ($name, $arr) = each %aux_dict)
        {
            $arr .= $aux_params->{$name};
        }
    }
    my @dtypes = map { $_->outputs->[0]->dtype } @exe_list;
    my $max_idx = pdl(map { $dtypes{$_} } @dtypes)->maximum_ind;
    my $gt = $ground_truth;
    if(not defined $gt)
    {
        $gt = { %{ $exe_list[$max_idx]->output_dict } };
        if($grad_req ne 'null')
        {
            %{$gt} = (%{$gt}, %{ $exe_list[$max_idx]->grad_dict });
        }
    }

    # test
    for my $exe (@exe_list)
    {
        $exe->forward(0);
    }
    enumerate(sub {
        my ($i, $exe) = @_;
        if($i == $max_idx)
        {
            return;
        }
        zip(sub {
            my ($name, $arr) = @_;
            my $gtarr = $gt->{$name}->astype($dtypes[$i])->aspdl;
            $arr = $arr->aspdl;
            Test::More::ok(
                almost_equal(
                    $arr, $gtarr,
                    $tol->{$dtypes[$i]}
                )
            );
        }, $output_names, $exe->outputs);
    }, \@exe_list);

    # train
    if ($grad_req ne 'null')
    {
        for my $exe (@exe_list)
        {
            $exe->forward(1);
            $exe->backward($exe->outputs);
        }
        enumerate(sub {
            my ($i, $exe) = @_;
            return if($i == $max_idx);
            zip(sub {
                my ($name, $arr) = @_;
                if (not defined $gt->{$name})
                {
                    Test::More::ok(not defined $arr);
                    return;
                }
                my $gtarr = $gt->{$name}->astype($dtypes[$i])->aspdl;
                $arr = $arr->aspdl;
                Test::More::ok(
                    almost_equal(
                        $arr, $gtarr,
                        $tol->{$dtypes[$i]}
                    )
                );
            }, [@$output_names, @$arg_names], [@{ $exe->outputs }, @{ $exe->grad_arrays }]);
        }, \@exe_list);
    }
    return $gt;
}

sub zip
{
    my ($sub, @arrays) = @_;
    my $len = @{ $arrays[0] };
    for (my $i = 0; $i < $len; $i++)
    {
        $sub->(map { $_->[$i] } @arrays);
    }
}

sub enumerate
{
    my ($sub, @arrays) = @_;
    my $len = @{ $arrays[0] };
    zip($sub, [0..$len-1], @arrays);
}

sub assert
{
    my ($input, $error_str) = @_;
    local($Carp::CarpLevel) = 1;
    Carp::confess($error_str//'AssertionError')
        unless $input;
}

=head2 same_array

    Check whether two NDArrays sharing the same memory block

    Parameters
    ----------

    array1 : NDArray
        First NDArray to be checked
    array2 : NDArray
        Second NDArray to be checked

    Returns
    -------
    bool
        Whether two NDArrays share the same memory
=cut

func same_array(
    AI::MXNet::NDArray $array1,
    AI::MXNet::NDArray $array2
)
{
    $array1 += 1;
    if(not same($array1->aspdl, $array2->aspdl))
    {
        $array1 -= 1;
        return 0
    }
    $array1 -= 1;
    return same($array1->aspdl, $array2->aspdl);
}

1;