# MXNet - Perl API

MXNet supports the Perl programming language. The MXNet Perl package brings flexible and efficient GPU
computing and state-of-art deep learning to Perl. It enables you to write seamless tensor/matrix computation with multiple GPUs in Perl.
It also lets you construct and customize the state-of-art deep learning models in Perl,
  and apply them to tasks, such as image classification and data science challenges.

One important thing to internalize is that Perl interface is written to be as close as possible to the Python's API,
so most if not all of Python's documentation and examples should just work in Perl after making few
changes in order to make the code a bit more Perlish. In nutshell just add $ sigils and replace . = \n with -> => ; and in 99% of cases
that's all that is needed there.
In addition please refer to [excellent metacpan doc interface](https://metacpan.org/release/AI-MXNet) and to very detailed
[MXNet Python API Documentation](http://mxnet.io/api/python/index.html).

AI::MXNet is seamlessly glued with PDL, the C++ level state can be easily initialized from PDL and the results can be
transferred to PDL objects in order to allow you to use all the glory and power of the PDL!

Here is how you can perform tensor or matrix computation in Perl with AI::MXNet and PDL:

```perl
pdl> use AI::MXNet qw(mx); # creates 'mx' module on the fly with the interface close to the Python's API

pdl> print $arr = mx->nd->ones([2, 3])
<AI::MXNet::NDArray 2x3 @cpu(0)>

pdl> print Data::Dumper::Dumper($arr->shape)
$VAR1 = [
          2,
          3
        ];

pdl> print (($arr*2)->aspdl) ## converts AI::MXNet::NDArray object to PDL object

[
 [2 2 2]
 [2 2 2]
]

pdl> print $arr = mx->nd->array([[1,2],[3,4]]) ## init the NDArray from Perl array ref given in PDL::pdl constructor format
<AI::MXNet::NDArray 2x2 @cpu(0)>
pdl> print $arr->aspdl

[
 [1 2]
 [3 4]
]

## init the NDArray from PDL but be aware that PDL methods expect the dimensions order in column major format
## AI::MXNet::NDArray is row major
pdl> print mx->nd->array(sequence(2,3))->aspdl ## 3 rows, 2 columns

[
 [0 1]
 [2 3]
 [4 5]
]
```
 ## Perl API Reference
 * [Module API](module.md) is a flexible high-level interface for training neural networks.
 * [Symbolic API](symbol.md) performs operations on NDArrays to assemble neural networks from layers.
 * [IO Data Loading API](io.md) performs parsing and data loading.
 * [NDArray API](ndarray.md) performs vector/matrix/tensor operations.
 * [KVStore API](kvstore.md) performs multi-GPU and multi-host distributed training.

