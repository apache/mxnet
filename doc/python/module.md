The base class of a modules. A module represents a computation component. The design
purpose of a module is that it abstract a computation "machine", that one can run forward,
backward, update parameters, etc. We aim to make the APIs easy to use, especially in the
case when we need to use imperative API to work with multiple modules (e.g. stochastic
depth network).

A module has several states:

- Initial state. Memory is not allocated yet, not ready for computation yet.
- Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated,
  ready for computation.
- Parameter initialized. For modules with parameters, doing computation before initializing
  the parameters might result in undefined outputs.
- Optimizer installed. An optimizer can be installed to a module. After this, the parameters
  of the module can be updated according to the optimizer after gradients are computed
  (forward-backward).

In order for a module to interactive with others, a module should be able to report the
following information (after binded).

- state information
    - `binded`: `bool`, indicating whether the memory buffers needed for computation
       has been allocated.
    - `for_training`: whether the module is binded for training (if binded).
    - `params_initialized`: `bool`, indicating whether the parameters of this modules
       has been initialized.
    - `optimizer_initialized`: 'bool`, indicating whether an optimizer is defined
       and initialized.
    - `inputs_need_grad`: `bool`, indicating whether gradients with respect to the
      input data is needed. Might be useful when implementing composition of modules.

- input/output information
    - `data_shapes`: a list of `(name, shape)`. In theory, since the memory is allocated,
      we could directly provide the data arrays. But in the case of data parallelization,
      the data arrays might not be of the same shape as viewed from the external world.
    - `label_shapes`: a list of `(name, shape)`. This might be `None` if the module does
      not need labels (e.g. it does not contains a loss function at the top).
    - `output_shapes`: a list of `(name, shape)` for outputs of the module.

- parameters (for modules with parameters)
    - `get_params()`: return a tuple `(arg_params, aux_params)`. Each of those
      is a dictionary of name to `NDArray` mapping. Those `NDArray` always lives on
      CPU. The actual parameters used for computing might live on other devices (GPUs),
      this function will retrieve (a copy of) the latest parameters. Therefore, modifying
    - `set_params(arg_params, aux_params)`: assign parameters to the devices
      doing the computation.
    - `init_params(...)`: a more flexible interface to assign or initialize the parameters.
    - `init_optimizer`: install optimizer for parameter updating.

- setup
    - `bind()`: prepare environment for computation.
    - `init_optimizer()`: install optimizer for parameter updating.

- computation
    - `forward(data_batch)`: forward operation.
    - `backward(out_grads=None)`: backward operation.
    - `update()`: update parameters according to installed optimizer.
    - `get_outputs()`: get outputs of the previous forward operation.
    - `get_input_grads()`: get the gradients with respect to the inputs computed
      in the previous backward operation.
    - `update_metric(metric)`: update performance metric for the previous forward
       computed results.

- other properties (mostly for backward compatability)
    - `symbol`: the underlying symbolic graph for this module (if any)
      This property is not necessarily constant. For example, for `BucketingModule`,
      this property is simply the *current* symbol being used. For other modules,
      this value might not be well defined.

When those intermediate-level API are implemented properly, the following
high-level API will be automatically available for a module:

- `fit`: train the module parameters on a data set
- `predict`: run prediction on a data set and collect outputs
- `score`: run prediction on a data set and evaluate performance
