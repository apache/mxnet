# Module API

## Overview

The module API, defined in the `module` (or simply `mod`) package (`AI::MXNet::Module` under the hood), provides an
intermediate and high-level interface for performing computation with a
`AI::MXNet::Symbol` or just `mx->sym`. One can roughly think a module is a machine which can execute a
program defined by a `Symbol`.

The class `AI::MXNet::Module` is a commonly used module, which accepts a `AI::MXNet::Symbol` as
the input:

```perl
pdl> $data = mx->symbol->Variable('data')
pdl> $fc1  = mx->symbol->FullyConnected($data, name=>'fc1', num_hidden=>128)
pdl> $act1 = mx->symbol->Activation($fc1, name=>'relu1', act_type=>"relu")
pdl> $fc2  = mx->symbol->FullyConnected($act1, name=>'fc2', num_hidden=>10)
pdl> $out  = mx->symbol->SoftmaxOutput($fc2, name => 'softmax')
pdl> $mod  = mx->mod->Module($out)  # create a module by given a Symbol
```

Assume there is a valid MXNet data iterator `data`. We can initialize the
module:

```perl
pdl> $mod->bind(data_shapes=>$data->provide_data,
         label_shapes=>$data->provide_label)  # create memory by given input shapes
pdl> $mod->init_params()  # initial parameters with the default random initializer
```

Now the module is able to compute. We can call high-level API to train and
predict:

```perl
pdl> $mod->fit($data, num_epoch=>10, ...)  # train
pdl> $mod->predict($new_data)  # predict on new data
```

or use intermediate APIs to perform step-by-step computations

```perl
pdl> $mod->forward($data_batch, is_train => 1)  # forward on the provided data batch
pdl> $mod->backward()  # backward to calculate the gradients
pdl> $mod->update()  # update parameters using the default optimizer
```
