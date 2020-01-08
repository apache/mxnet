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

# Data Providers

Data providers are wrappers that load external data, be it images, text, or general tensors,
and split it into mini-batches so that the model can consume the data in a uniformed way.

## AbstractDataProvider interface

```@docs
mx.AbstractDataProvider
```
 
The difference between *data* and *label* is that during training stage,
both *data* and *label* will be feeded into the model, while during
prediction stage, only *data* is loaded. Otherwise, they could be anything, with any names, and
of any shapes. The provided data and label names here should match the input names in a target
`SymbolicNode`.

A data provider should also implement the Julia iteration interface, in order to allow iterating
through the data set. The provider will be called in the following way:

```julia
for batch in eachbatch(provider)
    data = get_data(provider, batch)
end
```

which will be translated by Julia compiler into

```julia
state = Base.start(eachbatch(provider))
while !Base.done(provider, state)
    (batch, state) = Base.next(provider, state)
    data = get_data(provider, batch)
end
```
 
By default, `eachbatch` simply returns the provider itself, so the iterator interface
is implemented on the provider type itself. But the extra layer of abstraction allows us to
implement a data provider easily via a Julia `Task` coroutine. See the
data provider defined in [the char-lstm example](/api/julia/docs/api/tutorial/char-lstm/) for an example of using coroutine to define data
providers.

The detailed interface functions for the iterator API is listed below:

    Base.eltype(provider) -> AbstractDataBatch

Returns the specific subtype representing a data batch. See `AbstractDataBatch`.
* `provider::AbstractDataProvider`: the data provider.

    Base.start(provider) -> AbstractDataProviderState

This function is always called before iterating into the dataset. It should initialize
the iterator, reset the index, and do data shuffling if needed.
* `provider::AbstractDataProvider`: the data provider.

    Base.done(provider, state) -> Bool

True if there is no more data to iterate in this dataset.
* `provider::AbstractDataProvider`: the data provider.
* `state::AbstractDataProviderState`: the state returned by `Base.start` and `Base.next`.

    Base.next(provider) -> (AbstractDataBatch, AbstractDataProviderState)

Returns the current data batch, and the state for the next iteration.
* `provider::AbstractDataProvider`: the data provider.

Note sometimes you are wrapping an existing data iterator (e.g. the built-in libmxnet data iterator) that
is built with a different convention. It might be difficult to adapt to the interfaces stated here. In this
case, you can safely assume that

* `Base.start` will always be called, and called only once before the iteration starts.
* `Base.done` will always be called at the beginning of every iteration and always be called once.
* If `Base.done` return true, the iteration will stop, until the next round, again, starting with
  a call to `Base.start`.
* `Base.next` will always be called only once in each iteration. It will always be called after
  one and only one call to `Base.done`; but if `Base.done` returns true, `Base.next` will
  not be called.

With those assumptions, it will be relatively easy to adapt any existing iterator. See the implementation
of the built-in `MXDataProvider` for example.

!!! note
    Please do not use the one data provider simultaneously in two different places, either in parallel,
    or in a nested loop. For example, the behavior for the following code is undefined

    ```julia
    for batch in data
        # updating the parameters

        # now let's test the performance on the training set
        for b2 in data
            # ...
        end
    end
    ```

```@docs
mx.get_batch_size
mx.provide_data
mx.provide_label
```

## AbstractDataBatch interface

```@docs
mx.AbstractDataProviderState
mx.count_samples
mx.get_data
mx.get_label
mx.get
mx.load_data!
mx.load_label!
```

## Implemented providers and other methods

```@autodocs
Modules = [MXNet.mx]
Pages = ["io.jl"]
```
