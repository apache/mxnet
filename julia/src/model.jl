# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
    AbstractModel

The abstract super type of all models in MXNet.jl.
"""
abstract type AbstractModel end

"""
    FeedForward

The feedforward model provides convenient interface to train and predict on
feedforward architectures like multi-layer MLP, ConvNets, etc. There is no
explicitly handling of *time index*, but it is relatively easy to implement
unrolled RNN / LSTM under this framework (*TODO*: add example). For models
that handles sequential data explicitly, please use *TODO*...
"""
mutable struct FeedForward <: AbstractModel
  arch        :: SymbolicNode
  ctx         :: Vector{Context}

  arg_params  :: Dict{Symbol}
  aux_params  :: Dict{Symbol}

  pred_exec   :: Union{Executor,Void}

  # leave the rest fields undefined
  FeedForward(arch::SymbolicNode, ctx::Vector{Context}) = new(arch, ctx)
  FeedForward(arch::SymbolicNode, ctx::Context) = new(arch, [ctx])
end

"""
Get a split of `batch_size` into `n_split` pieces for data parallelization. Returns a vector
of length `n_split`, with each entry a `UnitRange{Int}` indicating the slice index for that
piece.
"""
function _split_inputs(batch_size::Int, n_split::Int)
  @assert(batch_size >= n_split)
  per_split = floor(Int, batch_size / n_split)
  counts    = Base.zeros(Int, n_split)+per_split
  extra     = batch_size - Base.sum(counts)
  counts[1:extra] += 1

  cum = [0, cumsum(counts)...]
  idx = [cum[i-1]+1:cum[i] for i = 2:length(cum)]
  return idx
end

"""
    FeedForward(arch :: SymbolicNode, ctx)

# Arguments:
* `arch`: the architecture of the network constructed using the symbolic API.
* `ctx`: the devices on which this model should do computation. It could be a single `Context`
         or a list of `Context` objects. In the latter case, data parallelization will be used
         for training. If no context is provided, the default context `cpu()` will be used.
"""
FeedForward(arch::SymbolicNode; context::Union{Context,Vector{Context}} = [cpu()]) =
  FeedForward(arch, context)

"""
    init_model(self, initializer; overwrite=false, input_shapes...)

Initialize the weights in the model.

This method will be called automatically when training a model. So there is usually no
need to call this method unless one needs to inspect a model with only randomly initialized
weights.

# Arguments:
* `self::FeedForward`: the model to be initialized.
* `initializer::AbstractInitializer`: an initializer describing how the weights should be initialized.
* `overwrite::Bool`: keyword argument, force initialization even when weights already exists.
* `input_shapes`: the shape of all data and label inputs to this model, given as keyword arguments.
                  For example, `data=(28,28,1,100), label=(100,)`.
"""
function init_model(self::FeedForward, initializer::AbstractInitializer; overwrite::Bool=false, input_shapes...)
  # all arg names, including data, label, and parameters
  arg_names    = list_arguments(self.arch)

  input_names  = [x[1] for x in input_shapes]

  param_names = setdiff(arg_names, input_names)
  aux_names   = list_auxiliary_states(self.arch)

  arg_shapes, out_shapes, aux_shapes = infer_shape(self.arch; input_shapes...)

  # If target dict is not yet defined set a temporary one
  if !isdefined(self, :arg_params)
    self.arg_params = Dict{Symbol, NDArray}()
  end
  if !isdefined(self, :aux_params)
    self.aux_params = Dict{Symbol, NDArray}()
  end

  arg_params = Dict{Symbol,NDArray}()
  aux_params = Dict{Symbol,NDArray}()

  for (name, shape) in filter(x -> in(x[1],param_names), zip(arg_names, arg_shapes))
    if haskey(self.arg_params, name)
      if shape == size(self.arg_params[name])
        arg_params[name] = self.arg_params[name]
        continue
      else
        warn("Shape mismatch for $name. Overwriting with new one.")
        delete!(self.arg_params, name)
      end
    end
    arg_params[name] = empty(shape)
  end

  for (name, shape) in zip(aux_names, aux_shapes)
    if haskey(self.aux_params, name)
      if shape == size(self.aux_params[name])
        aux_params[name] = self.aux_params[name]
        continue
      else
        warn("Shape mismatch for $name. Overwriting with new one.")
        delete!(self.aux_params, name)
      end
    end
    aux_params[name] = empty(shape)
  end

  for (k,v) in arg_params
    if overwrite || !haskey(self.arg_params, k)
      init(initializer, k, v)
    end
  end
  for (k,v) in aux_params
    if overwrite || !haskey(self.aux_params, k)
      init(initializer, k, v)
    end
  end

  self.arg_params = arg_params
  self.aux_params = aux_params

  return (arg_names, param_names, aux_names)
end

function _setup_predictor(self::FeedForward, overwrite::Bool=false; verbosity::Integer = 1, data_shapes...)
  if !isdefined(self, :pred_exec) || isa(self.pred_exec, Void) || overwrite
    if !isdefined(self, :arg_params) || !isdefined(self, :aux_params)
      @assert(false, "Model weights not defined, please init or train the model, or load from file")
    end

    # the predictor use only the first device
    self.pred_exec = simple_bind(self.arch, self.ctx[1]; grad_req=GRAD_NOP, data_shapes...)
    dbg_str = mx.debug_str(self.pred_exec)
    verbosity >= 1 && info(string("TempSpace: ", split(dbg_str, ['\n'])[end-2]..., " on ", self.ctx[1]))
    copy_params_from(self.pred_exec, self.arg_params, self.aux_params)
  else
    # make sure the new setup is compatible with the existing one
    for (d_name, d_shape) in data_shapes
      @assert(d_shape == size(self.pred_exec.arg_dict[d_name]),
              "Shape of $d_name mismatch with existing predictor, use overwrite=true overwrite existing predictor")
    end
  end
end

"""
    predict(self, data; overwrite=false, callback=nothing)

Predict using an existing model. The model should be already initialized, or trained or loaded from
a checkpoint. There is an overloaded function that allows to pass the callback as the first argument,
so it is possible to do

```julia
predict(model, data) do batch_output
  # consume or write batch_output to file
end
```

# Arguments:
* `self::FeedForward`:  the model.
* `data::AbstractDataProvider`: the data to perform prediction on.
* `overwrite::Bool`: an `Executor` is initialized the first time predict is called. The memory
                     allocation of the `Executor` depends on the mini-batch size of the test
                     data provider. If you call predict twice with data provider of the same batch-size,
                     then the executor can be potentially be re-used. So, if `overwrite` is false,
                     we will try to re-use, and raise an error if batch-size changed. If `overwrite`
                     is true (the default), a new `Executor` will be created to replace the old one.
* `verbosity::Integer`: Determines the verbosity of the print messages. Higher numbers
          leads to more verbose printing. Acceptable values are
          - `0`: Do not print anything during prediction
          - `1`: Print allocation information during prediction

!!! note
    Prediction is computationally much less costly than training, so the bottleneck sometimes becomes the IO
    for copying mini-batches of data. Since there is no concern about convergence in prediction, it is better
    to set the mini-batch size as large as possible (limited by your device memory) if prediction speed is a
    concern.

    For the same reason, currently prediction will only use the first device even if multiple devices are
    provided to construct the model.

!!! note
    If you perform further after prediction. The weights are not automatically synchronized if `overwrite`
    is set to false and the old predictor is re-used. In this case
    setting `overwrite` to true (the default) will re-initialize the predictor the next time you call
    predict and synchronize the weights again.

See also [`train`](@ref), [`fit`](@ref), [`init_model`](@ref), and [`load_checkpoint`](@ref)
"""
function predict(callback::Function, self::FeedForward, data::AbstractDataProvider;
                 overwrite::Bool = true, verbosity::Integer = 1)
  predict(self, data; overwrite = overwrite, callback=callback, verbosity = verbosity)
end
function predict(self::FeedForward, data::AbstractDataProvider;
                 overwrite::Bool = true, callback::Union{Function,Void}=nothing, verbosity::Integer = 1)
  data_shapes = provide_data(data)
  data_names  = [x[1] for x in data_shapes]
  _setup_predictor(self, overwrite; verbosity = verbosity, data_shapes...)

  batch_size  = get_batch_size(data)
  data_arrays =  [self.pred_exec.arg_dict[name] for name in data_names]
  output_list = [Array{MX_float}[] for i=1:length(self.pred_exec.outputs)]
  for batch in eachbatch(data)
    load_data!(data, batch, data_arrays)
    forward(self.pred_exec, is_train=false)
    if isa(callback, Void)
      # no callback, accumulate the data and return at the end
      for (o_list, o_nd) in zip(output_list, self.pred_exec.outputs)
        push!(o_list, copy(slice(o_nd, 1:count_samples(data, batch))))
      end
    else
      outputs = self.pred_exec.outputs
      if length(outputs) == 1
        outputs = outputs[1]
      end
      callback(outputs)
    end
  end

  if !isa(callback, Void)
    # callback exists, do not accumulate data
    return nothing
  end

  if isempty(output_list)
    # maybe model does not have outputs
    return nothing
  end
  if isempty(output_list[1])
    # maybe no output because data is empty
    return length(output_list) == 1 ? output_list[1] : output_list
  end

  # concatenate along mini-batches
  output_arrays = [cat(ndims(x[1]), x...) for x in output_list]
  if length(output_arrays) == 1
    # only 1 output, return it directly, instead of a list
    output_arrays = output_arrays[1]
  end
  return output_arrays
end

function _init_model(self::FeedForward, data::AbstractDataProvider,
                     initializer::AbstractInitializer, overwrite::Bool)
  init_model(self, initializer; overwrite=overwrite,
             [provide_data(data)..., provide_label(data)...]...)
end

function _create_kvstore(kv_type::Symbol, num_device::Int, arg_params::Dict{Symbol}, verbosity::Int)
  if num_device == 1 && !ismatch(r"dist", string(kv_type))
    return nothing
  else
    if kv_type == :local
      max_size = maximum([prod(size(param)) for (k,param) in arg_params])
      if max_size < 1024 * 1024 * 16
        kv_type = :local_update_cpu
      else
        kv_type = :local_allreduce_cpu
      end
      verbosity >= 2 && info("Auto-select kvstore type = $kv_type")
    end
    return KVStore(kv_type)
  end
end

@defstruct TrainingOptions (
  initializer :: AbstractInitializer = UniformInitializer(0.01),
  n_epoch     :: Int = 10,
  eval_data   :: Union{Void,AbstractDataProvider} = nothing,
  eval_metric :: AbstractEvalMetric = Accuracy(),
  kvstore     :: Union{Symbol,KVStore} = :local,
  force_init  :: Bool = false,
  callbacks   :: Vector{AbstractCallback} = AbstractCallback[],
  verbosity   :: Int = 3,
  η_decay     :: Symbol = :epoch,
)

function _invoke_callbacks(m::FeedForward, callbacks::Vector{AbstractCallback},
                           state::OptimizationState, type_filter::Type;
                           metric = Vector{Tuple{Symbol,Real}}())
  map(callbacks) do cb
    !isa(cb, type_filter) && return

    # epoch callback have extra access to the model object
    type_filter == AbstractEpochCallback && return cb(m, state, metric)

    cb(state)
  end
end

"""
    train(model :: FeedForward, ...)

Alias to [`fit`](@ref).
"""
train(m::FeedForward, opt::AbstractOptimizer, data::AbstractDataProvider; kw...) =
  fit(m, opt, data; kw...)

"""
    fit(model::FeedForward, optimizer, data; kwargs...)

Train the `model` on `data` with the `optimizer`.

* `model::FeedForward`: the model to be trained.
* `optimizer::AbstractOptimizer`: the optimization algorithm to use.
* `data::AbstractDataProvider`: the training data provider.
* `n_epoch::Int`: default 10, the number of full data-passes to run.
* `eval_data::AbstractDataProvider`: keyword argument, default `nothing`. The data provider for
          the validation set.
* `eval_metric::AbstractEvalMetric`: keyword argument, default [`Accuracy()`](@ref). The metric used
          to evaluate the training performance. If `eval_data` is provided, the same metric is also
          calculated on the validation set.
* `kvstore`: keyword argument, default `:local`. The key-value store used to synchronize gradients
          and parameters when multiple devices are used for training.
   :type kvstore: `KVStore` or `Symbol`
* `initializer::AbstractInitializer`: keyword argument, default `UniformInitializer(0.01)`.
* `force_init::Bool`: keyword argument, default false. By default, the random initialization using the
          provided `initializer` will be skipped if the model weights already exists, maybe from a previous
          call to [`train`](@ref) or an explicit call to [`init_model`](@ref) or [`load_checkpoint`](@ref). When
          this option is set, it will always do random initialization at the begining of training.
* `callbacks::Vector{AbstractCallback}`: keyword argument, default `[]`. Callbacks to be invoked at each epoch or mini-batch,
          see `AbstractCallback`.
* `verbosity::Int`: Determines the verbosity of the print messages. Higher numbers
          leads to more verbose printing. Acceptable values are
          - `0`: Do not print anything during training
          - `1`: Print starting and final messages
          - `2`: Print one time messages and a message at the start of each epoch
          - `3`: Print a summary of the training and validation accuracy for each epoch
* `η_decay::Symbol`: `:epoch` or `:batch`, decay learning rate on epoch or batch.
"""
function fit(self::FeedForward, optimizer::AbstractOptimizer, data::AbstractDataProvider;
             kwargs...)
  opts = TrainingOptions(; kwargs...)

  opts.verbosity >= 1 && info("Start training on $(self.ctx)")

  batch_size  = get_batch_size(data)
  num_dev     = length(self.ctx)
  slices      = _split_inputs(batch_size, num_dev)

  # initialize parameters
  opts.verbosity >= 2 && info("Initializing parameters...")
  arg_names, param_names, aux_names = _init_model(self, data, opts.initializer, opts.force_init)

  # setup kvstore
  kvstore = opts.kvstore
  if isa(kvstore, Symbol)
    opts.verbosity >= 2 && info("Creating KVStore...")
    kvstore = _create_kvstore(kvstore, length(self.ctx), self.arg_params, opts.verbosity)
  end

  update_on_kvstore = true
  if isa(kvstore, Void) || ismatch(r"local_allreduce", string(get_type(kvstore)))
    update_on_kvstore = false
  end

  # get grad attribute to allow for freezing
  freeze_names = Symbol[]
  for (attr, value) in list_all_attr(self.arch)
    sattr = string(attr)
    if endswith(sattr, "grad") && value == "freeze"
      push!(freeze_names, Symbol(sattr[1:end-5]))
    end
  end
  # Needs to correspond to the correct id in the update loop layer idx=1:length(param_names).
  freeze_idx = filter(i -> in(param_names[i], freeze_names), 1:length(param_names))

  # Setup grad_req as a dictionary
  grad_req = Dict{Symbol,GRAD_REQ}()
  for param in param_names
    if in(param, freeze_names)
      grad_req[param] = GRAD_NOP
    else
      grad_req[param] = GRAD_WRITE
    end
  end

  train_execs = Array{Executor}(num_dev)
  for i = 1:num_dev
    data_shapes = Dict(map((x) -> x[1] => tuple(x[2][1:end-1]...,length(slices[i])), provide_data(data)))
    label_shapes = Dict(map((x) -> x[1] => tuple(x[2][1:end-1]...,length(slices[i])), provide_label(data)))
    train_execs[i] = simple_bind(self.arch, self.ctx[i]; grad_req=grad_req, data_shapes..., label_shapes...)
    dbg_str = mx.debug_str(train_execs[i])
    opts.verbosity >= 2 && info(string("TempSpace: ", split(dbg_str, ['\n'])[end-2]..., " on ", self.ctx[i]))

    copy_params_from(train_execs[i], self.arg_params, self.aux_params)
  end

  # set up input data structures
  data_names   = [x[1] for x in provide_data(data)]
  label_names  = [x[1] for x in provide_label(data)]

  data_arrays  = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                  for name in data_names]
  label_arrays = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                  for name in label_names]

  param_idx    = filter(i -> in(arg_names[i], param_names), 1:length(arg_names))

  param_arrays = [NDArray[exec.arg_arrays[i] for exec in train_execs] for i in param_idx]
  grad_arrays  = [NDArray[exec.grad_arrays[i] for exec in train_execs] for i in param_idx]
  aux_arrays   = [NDArray[exec.aux_arrays[i] for exec in train_execs] for i = 1:length(aux_names)]

  op_state = OptimizationState(batch_size)
  # set up the gradient rescaling if user not set
  iszero(optimizer.scale) && (optimizer.scale = 1 / batch_size)

  if !update_on_kvstore
    updater = getupdater(optimizer)
  end

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

    opts.verbosity >= 2 && info("Initializing KVStore...")
    # init kv with gradients
    for idx = 1:length(param_arrays)
      param_on_devs = param_arrays[idx]

      init!(kvstore, idx, self.arg_params[param_names[idx]])

      if update_on_kvstore
        # pull weights back
        pull!(kvstore, idx, param_on_devs, priority=-idx)
      end
    end
  end

  # set up output and labels in CPU for evaluation metric
  output_shapes = [tuple(size(x)[1:end-1]...,batch_size) for x in train_execs[1].outputs]
  cpu_dev = Context(CPU)
  cpu_output_arrays = [empty(shape, cpu_dev) for shape in output_shapes]
  cpu_label_arrays  = [empty(shape, cpu_dev) for (name,shape) in provide_label(data)]

  # invoke callbacks on epoch 0
  _invoke_callbacks(self, opts.callbacks, op_state, AbstractEpochCallback)

  opts.verbosity >= 2 && info("Start training...")
  for i_epoch = 1:opts.n_epoch
    time_start = time()
    reset!(opts.eval_metric)

    op_state.curr_epoch = i_epoch
    op_state.curr_batch = 0

    # invoke callbacks on iteration 0
    _invoke_callbacks(self, opts.callbacks, op_state, AbstractBatchCallback)

    for batch in eachbatch(data)
      load_data!(data, batch, data_arrays)
      load_label!(data, batch, label_arrays)

      # forward and backward
      for (texec, islice) in zip(train_execs, slices)
        forward(texec, is_train=true)

        # copy outputs into cpu ndarray, for evaluation metric
        for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
          copy!(slice(cpu_out, islice), dev_out)
        end

        backward(texec)
      end

      op_state.curr_iter  += 1
      op_state.curr_batch += 1

      # update parameters
      for idx = 1:length(param_names)
        if in(idx, freeze_idx)
          continue # Skip parameter update entirely
        end

        # gradient synchronization
        if !isa(kvstore, Void)
          # push gradient, priority is negative index
          push!(kvstore, idx, grad_arrays[idx], priority=-idx)
          if update_on_kvstore
            # pull back the weights
            pull!(kvstore, idx, param_arrays[idx], priority=-idx)
          else
            # pull back the sum-ed gradients, to the same locations
            pull!(kvstore, idx, grad_arrays[idx], priority=-idx)
          end
        end

        if !update_on_kvstore
          # manual updating
          for i_dev = 1:num_dev
            # create a fake index, so that the updater create states
            # for different param AND different devices, TODO(mli)
            # use a better solution later
            fake_idx = idx * num_dev + i_dev
            updater(fake_idx, grad_arrays[idx][i_dev], param_arrays[idx][i_dev])
          end
        end
      end

      # trigger learning rate decay
      opts.η_decay == :batch && update!(optimizer.η_sched)

      # invoke callbacks after finishing each iteration
      _invoke_callbacks(self, opts.callbacks, op_state, AbstractBatchCallback)

      # update evaluation metric on training set
      load_label!(data, batch, cpu_label_arrays)
      update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
    end # end of one epoch

    time_stop = time()
    metric = get(opts.eval_metric)
    opts.verbosity >= 2 && info(format("== Epoch {1:0>3d}/{2:0>3d} ==========", i_epoch, opts.n_epoch))
    if opts.verbosity >= 3
        info("## Training summary")
        for (name, value) in metric
            info(format("{1:>18s} = {2:.4f}", string(name), value))
        end
        info(format("{1:>18s} = {2:.4f} seconds", "time", time_stop-time_start))
    end

    # evaluation on validation set
    if !isa(opts.eval_data, Void)
      # because we are re-using the memory allocated for the training network,
      # the batch_size of the validation dataset must be the same as the training
      # batch_size
      @assert(get_batch_size(opts.eval_data) == batch_size)

      reset!(opts.eval_metric)
      for batch in eachbatch(opts.eval_data)
        load_data!(opts.eval_data, batch, data_arrays)

        # forward and backward
        for (texec, islice) in zip(train_execs, slices)
          forward(texec, is_train=true)

          # copy outputs into cpu ndarray, for evaluation metric
          for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
            copy!(slice(cpu_out, islice), dev_out)
          end
        end
        load_label!(opts.eval_data, batch, cpu_label_arrays)
        update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
      end

      if opts.verbosity >= 3
          info("## Validation summary")
          for (name, value) in get(opts.eval_metric)
            info(format("{1:>18s} = {2:.4f}", string(name), value))
          end
      end
    end

    if i_epoch == opts.n_epoch || any(x->isa(x, AbstractEpochCallback), opts.callbacks)
      # copy data back to cpu
      for (name, weights) in zip(param_names, param_arrays)
        # average parameters across devices
        weight = +([copy(w, cpu()) for w in weights]...) / length(weights)
        copy!(self.arg_params[name], weight)
      end
      for (name, aux_devs) in zip(aux_names, aux_arrays)
        aux_avg = +([copy(aux, cpu()) for aux in aux_devs]...) / length(aux_devs)
        copy!(self.aux_params[name], aux_avg)
      end
    end

    # trigger learning rate decay
    opts.η_decay == :epoch && update!(optimizer.η_sched)

    _invoke_callbacks(self, opts.callbacks, op_state, AbstractEpochCallback; metric=metric)
  end # end of all epochs

  opts.verbosity >= 1 && info("Finish training on $(self.ctx)")
  nothing
end

save_checkpoint(self::FeedForward, prefix::AbstractString, state::OptimizationState) =
  save_checkpoint(self.arch, self.arg_params, self.aux_params, prefix, state.curr_epoch)

function save_checkpoint(sym::SymbolicNode, arg_params::Dict{Symbol},
                         aux_params::Dict{Symbol}, prefix::AbstractString, epoch::Int)
  save("$prefix-symbol.json", sym)
  save_dict = Dict{Symbol, NDArray}(map((x) -> Symbol("arg:$(x[1])") => x[2], arg_params))
  if !isempty(aux_params)
    merge!(save_dict, Dict(map((x) -> Symbol("aux:$(x[1])") => x[2], aux_params)))
  end
  save_filename = format("{1}-{2:04d}.params", prefix, epoch)
  save(save_filename, save_dict)
  info("Saved checkpoint to '$save_filename'")
end

function load_checkpoint(prefix::AbstractString, epoch::Int)
  arch       = load("$prefix-symbol.json", SymbolicNode)
  saved_dict = load(format("{1}-{2:04d}.params", prefix, epoch), NDArray)
  arg_params = Dict{Symbol,Any}()
  aux_params = Dict{Symbol,Any}()
  for (k,v) in saved_dict
    tp, name = split(string(k), ':')
    name = Symbol(name)
    if tp == "arg"
      arg_params[name] = v
    else
      aux_params[name] = v
    end
  end

  return (arch, arg_params, aux_params)
end

"""
    load_checkpoint(prefix, epoch, ::mx.FeedForward; context)

Load a mx.FeedForward model from the checkpoint *prefix*, *epoch* and optionally provide a context.
"""
function load_checkpoint(prefix::AbstractString, epoch::Int, ::Type{FeedForward}; context = nothing)
  arch, arg_params, aux_params = load_checkpoint(prefix, epoch)
  model = FeedForward(arch, context = context)
  model.arg_params = arg_params
  model.aux_params = aux_params
  return model
end

function load_checkpoint(self::FeedForward, prefix::AbstractString, epoch::Int;
                         overwrite::Bool = true, allow_different_arch::Bool = false)
  if isdefined(self, :arg_params) && isdefined(self, :aux_params) && !overwrite
    info("model weights already exists, skip loading... (call with overwrite=true if needed)")
    return self
  end

  arch, arg_params, aux_params = load_checkpoint(prefix, epoch)
  if !allow_different_arch
    # TODO: is there better way to compare two symbols
    @assert(to_json(self.arch) == to_json(arch), "Cannot load from a checkpoint with different network architecture")
  end
  self.arg_params = arg_params
  self.aux_params = aux_params
  return self
end
