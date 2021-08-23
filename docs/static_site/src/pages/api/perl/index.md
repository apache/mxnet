---
layout: page_api
title: Perl Guide
action: Get Started
action_url: /get_started
permalink: /api/perl
tag: perl
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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
[MXNet Python API Documentation]({{'/api/python' | relative_url}}).

AI::MXNet supports new imperative PyTorch like Gluon MXNet interface. Please get acquainted with this new interface
at [Dive into Deep Learning](https://www.d2l.ai/).

For specific Perl Gluon usage please refer to Perl examples and tests directories on github, but be assured that the Python and Perl usage
are extremely close in order to make the use of the Python Gluon docs and examples as easy as possible.

AI::MXNet is seamlessly glued with [PDL](https://metacpan.org/release/PDL), the C++ level state can be easily initialized from PDL and the results can be
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

Export/import to/from sparse MXNet tensors are supported via [PDL::CCS](https://metacpan.org/release/PDL-CCS).
Please check out the examples directory for the examples on how to use the sparse matrices.
