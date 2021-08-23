---
layout: page_api
title: Symbol API
is_tutorial: true
tag: perl
permalink: /api/perl/docs/tutorials/symbol
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

# MXNet Perl Symbolic API

Topics:

* [How to Compose Symbols](#how-to-compose-symbols) introduces operator overloading of symbols.
* [Symbol Attributes](#symbol-attributes) describes how to attach attributes to symbols.
* [Serialization](#serialization) explains how to save and load symbols.
* [Executing Symbols](#executing-symbols) explains how to evaluate the symbols with data.
* [Multiple Outputs](#multiple-outputs) explains how to configure multiple outputs.

## How to Compose Symbols

The symbolic API provides a way to configure computation graphs.
You can configure the graphs either at the level of neural network layer operations or as fine-grained operations.


The basic arithmetic operators (plus, minus, div, multiplication) are overloaded for
*element-wise operations* of symbols.

The following example creates a computation graph that adds two inputs together.

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a =  mx->symbol->Variable("a")
pdl> $b =  mx->symbol->Variable("b")
pdl> $c = $a + $b
```

## Symbol Attributes

You can add an attribute to a symbol by providing an attribute hash when you create a symbol.

```perl
$data =  mx->symbol->Variable("data", attr => { mood => "angry" })
$op   =  mx->symbol->Convolution(data => $data, kernel => [1, 1], num_filter => 1, attr => { mood => "so so" })
```

For proper communication with the C++ backend, both the key and values of the attribute dictionary should be strings. To retrieve the attributes, use `->attr($key)`:

```
    $data->attr("mood")
```

To attach attributes, you can use ```AI::MXNet::AttrScope```. ```AI::MXNet::AttrScopeAttrScope``` automatically adds
the specified attributes to all of the symbols created within that scope.
The user can also inherit this object to change naming behavior. For example:

```perl
use AI::MXNet qw(mx);
use Test::More tests => 3;
my ($data, $gdata);
{
    local($mx::AttrScope) = mx->AttrScope(group=>4, data=>'great');
    $data = mx->sym->Variable("data", attr => { dtype => "data", group => "1" });
    $gdata = mx->sym->Variable("data2");
}
ok($gdata->attr("group") == 4);
ok($data->attr("group") == 1);

my $exceedScopeData = mx->sym->Variable("data3");
ok((not defined $exceedScopeData->attr("group")), "No group attr in global attr scope");
```

## Serialization

There are two ways to save and load the symbols. You can use the `mx->symbol->save` and `mxnet->symbol->load` functions to serialize the ```AI::MXNet::Symbol``` objects.
The advantage of using `save` and `load` functions is that it is language agnostic and cloud friendly.
The symbol is saved in JSON format. You can also get a JSON string directly using `$symbol->tojson`.

The following example shows how to save a symbol to an S3 bucket, load it back, and compare two symbols using a JSON string.

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a = mx->sym->Variable("a")
pdl> $b = mx->sym->Variable("b")
pdl> $c = $a + $b
pdl> $c->save("s3://my-bucket/symbol-c.json")
pdl> $c2 = $c->load("s3://my-bucket/symbol-c.json")
pdl> ok($c->tojson eq $c2->tojson)
ok 1
```

## Executing Symbols

After you have assembled a set of symbols into a computation graph, the MXNet engine can evaluate them.
If you are training a neural network, this is typically
handled by the high-level [AI::MXNet::Module package](module) and the [`fit()`] function.

For neural networks used in "feed-forward", "prediction", or "inference" mode (all terms for the same
thing: running a trained network), the input arguments are the
input data, and the weights of the neural network that were learned during training.

To manually execute a set of symbols, you need to create an [`AI::MXNet::Executor`] object,
which is typically constructed by calling the [`simple_bind(<parameters>)`] method on a AI::MXNet::Symbol.
