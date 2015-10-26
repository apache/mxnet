abstract AbstractModel

type FeedForward <: AbstractModel
  arch        :: Symbol
  ctx         :: Vector{Context}

  arg_params  :: Dict{Base.Symbol, NDArray}
  aux_params  :: Dict{Base.Symbol, NDArray}

  pred_exec   :: Union{Executor, Void}

  # leave the rest fields undefined
  FeedForward(arch :: Symbol, ctx :: Vector{Context}) = new(arch, ctx)
end

"""Get a split of `batch_size` into `n_split` pieces for data parallelization. Returns a vector
    of length `n_split`, with each entry a `UnitRange{Int}` indicating the slice index for that
    piece.
"""
function _split_inputs(batch_size :: Int, n_split :: Int)
  @assert(batch_size >= n_split)
  per_split = floor(Int, batch_size / n_split)
  counts    = Base.zeros(Int, n_split)+per_split
  extra     = batch_size - sum(counts)
  counts[1:extra] += 1

  cum = [0, cumsum(counts)...]
  idx = [cum[i-1]+1:cum[i] for i = 2:length(cum)]
  return idx
end

function FeedForward(arch :: Symbol; context :: Union{Context, Vector{Context}, Void} = nothing)
  if isa(context, Void)
    context = [Context(CPU)]
  elseif isa(context, Context)
    context = [context]
  end
  FeedForward(arch, context)
end

"""Initialize the weights in the model.

This method will be called automatically when training a model. So there is usually no
need to call this method unless one needs to inspect a model with only randomly initialized
weights.

**Parameters**

* `self`: the model to be initialized
* `initializer`: an `AbstractInitializer`
* `input_shapes`: the shape of all data and label inputs to this model, given as keyword arguments.
"""
function init_model(self :: FeedForward, initializer :: AbstractInitializer; input_shapes...)
  # all arg names, including data, label, and parameters
  arg_names    = list_arguments(self.arch)

  input_names  = [x[1] for x in input_shapes]

  param_names = setdiff(arg_names, input_names)
  aux_names   = list_auxiliary_states(self.arch)

  arg_shapes, out_shapes, aux_shapes = infer_shape(self.arch; input_shapes...)
  if !isdefined(self, :arg_params)
    param_name_shapes = filter(x -> in(x[1],param_names), zip(arg_names, arg_shapes))
    self.arg_params = Dict([name => empty(shape) for (name,shape) in param_name_shapes])
  end
  if !isdefined(self, :aux_params)
    self.aux_params = Dict([name => empty(shape) for (name,shape) in zip(aux_names,aux_shapes)])
  end

  # initialize the contents of the parameters
  for (k,v) in self.arg_params
    initializer(k, v)
  end
  for (k,v) in self.aux_params
    initializer(k, v)
  end

  return (arg_names, param_names, aux_names)
end

function _init_model(self :: FeedForward, data :: AbstractDataProvider, initializer :: AbstractInitializer)
  init_model(self, initializer; [provide_data(data)..., provide_label(data)...]...)
end

function _create_kvstore(kv_type :: Base.Symbol, num_device :: Int, arg_params :: Dict{Base.Symbol,NDArray})
  if num_device == 1 && !ismatch(r"dist", string(kv_type))
    kv = nothing
  else
    if kv_type == :local
      max_size = maximum([prod(size(param)) for (k,param) in arg_params])
      if max_size < 1024 * 1024 * 16
        kv_type = :local_update_cpu
      else
        kv_type = :local_allreduce_cpu
      end
      info("Auto-select kvstore type = $kv_type")
    end
    kv = KVStore(kv_type)
  end

  update_on_kvstore = true
  if isa(kv, Void) || ismatch(r"local_allreduce", string(get_type(kv)))
    update_on_kvstore = false
  end

  return (kv, update_on_kvstore)
end

@defstruct TrainingOptions Any (
  initializer :: AbstractInitializer = UniformInitializer(0.01),
  n_epoch     :: Int = 10,
  eval_data   :: Union{Void, AbstractDataProvider} = nothing,
  eval_metric :: AbstractEvalMetric = Accuracy(),
  kvstore     :: Union{Base.Symbol, KVStore} = :local,
  callbacks   :: Vector{AbstractCallback} = AbstractCallback[],
)

function _invoke_callbacks(self::FeedForward, callbacks::Vector{AbstractCallback}, param::CallbackParams, type_filter::Type)
  map(callbacks) do cb
    if isa(cb, type_filter)
      if type_filter == AbstractEpochCallback
        # epoch callback have extra access to the model object
        cb(self, param)
      else
        cb(param)
      end
    end
  end
end

function _setup_predictor(self :: FeedForward, overwrite :: Bool=false; data_shapes...)
  if !isdefined(self, :pred_exec) || isa(self.pred_exec, Void) || overwrite
    if !isdefined(self, :arg_params) || !isdefined(self, :aux_params)
      @assert(false, "Model weights not defined, please init or train the model, or load from file")
    end
  else
    # make sure the new setup is compatible with the existing one
    for (d_name, d_shape) in data_shapes
      @assert(d_shape == size(self.pred_exec.arg_dict[d_name]),
              "Shape of $d_name mismatch with existing predictor, use overwrite=true overwrite existing predictor")
    end
  end
end

function predict(self :: FeedForward, data :: AbstractDataProvider)
end

function train(self :: FeedForward, optimizer :: AbstractOptimizer, data :: AbstractDataProvider; kwargs...)
  fit(self, optimizer, data; kwargs...)
end
function fit(self :: FeedForward, optimizer :: AbstractOptimizer, data :: AbstractDataProvider; kwargs...)
  opts = TrainingOptions(; kwargs...)

  info("Start training on $(self.ctx)")

  batch_size  = get_batch_size(data)
  num_dev     = length(self.ctx)
  slices      = _split_inputs(batch_size, num_dev)

  # initialize parameters
  info("Initializing parameters...")
  arg_names, param_names, aux_names = _init_model(self, data, opts.initializer)

  # setup kvstore
  kvstore = opts.kvstore
  if isa(kvstore, Base.Symbol)
    info("Creating KVStore...")
    kvstore, update_on_kvstore = _create_kvstore(kvstore, length(self.ctx), self.arg_params)
  end

  train_execs = Array(Executor, num_dev)
  for i = 1:num_dev
    data_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_data(data)]
    label_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_label(data)]
    train_execs[i] = simple_bind(self.arch, self.ctx[i]; grad_req=GRAD_WRITE, data_shapes..., label_shapes...)

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

  optimizer.batch_size = batch_size
  cb_param = CallbackParams(batch_size)

  if !update_on_kvstore
    updater = get_updater(optimizer)
  end

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

    info("Initializing KVStore...")
    # init kv with gradients
    for idx = 1:length(param_arrays)
      param_on_devs = param_arrays[idx]
      grad_on_devs  = grad_arrays[idx]

      init!(kvstore, idx, self.arg_params[param_names[idx]])

      # pull weights back
      pull!(kvstore, idx, param_on_devs, priority=-idx)
    end
  end

  # set up output and labels in CPU for evaluation metric
  output_shapes = [tuple(size(x)[1:end-1]...,batch_size) for x in train_execs[1].outputs]
  cpu_dev = Context(CPU)
  cpu_output_arrays = [empty(shape, cpu_dev) for shape in output_shapes]
  cpu_label_arrays  = [empty(shape, cpu_dev) for (name,shape) in provide_label(data)]
  cpu_label_arrays_full_slice = [SlicedNDArray[(1:batch_size, x)] for x in cpu_label_arrays]

  # invoke callbacks on epoch 0
  _invoke_callbacks(self, opts.callbacks, cb_param, AbstractEpochCallback)

  # now start training...
  for i_epoch = 1:opts.n_epoch
    time_start = time()
    reset!(opts.eval_metric)

    cb_param.curr_epoch = i_epoch
    cb_param.curr_iter = 0

    # invoke callbacks on iteration 0
    _invoke_callbacks(self, opts.callbacks, cb_param, AbstractIterationCallback)

    for batch in data
      load_data!(batch, data_arrays)
      load_label!(batch, label_arrays)

      # forward and backward
      for (texec, islice) in zip(train_execs, slices)
        forward(texec, is_train=true)

        # copy outputs into cpu ndarray, for evaluation metric
        for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
          copy!(slice(cpu_out, islice), dev_out)
        end

        backward(texec)
      end

      # update parameters
      for idx = 1:length(param_names)
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

      # invoke callbacks after finishing each iteration
      _invoke_callbacks(self, opts.callbacks, cb_param, AbstractIterationCallback)
      cb_param.curr_iter += 1

      # update evaluation metric on training set
      load_label!(batch, cpu_label_arrays_full_slice)
      update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
    end # end of one epoch

    time_stop = time()
    info(format("== Epoch {1:0>3d} ==========", i_epoch))
    info("## Training summary")
    for (name, value) in get(opts.eval_metric)
      info(format("{1:>15s} = {2:.4f}", name, value))
    end
    info(format("{1:>15s} = {2:.4f} seconds", "time", time_stop-time_start))

    # evaluation on validation set
    if !isa(opts.eval_data, Void)
      # because we are re-using the memory allocated for the training network,
      # the batch_size of the validation dataset must be the same as the training
      # batch_size
      @assert(get_batch_size(opts.eval_data) == batch_size)

      reset!(opts.eval_metric)
      for batch in opts.eval_data
        load_data!(batch, data_arrays)

        # forward and backward
        for (texec, islice) in zip(train_execs, slices)
          forward(texec, is_train=true)

          # copy outputs into cpu ndarray, for evaluation metric
          for (cpu_out, dev_out) in zip(cpu_output_arrays, texec.outputs)
            copy!(slice(cpu_out, islice), dev_out)
          end
        end
        load_label!(batch, cpu_label_arrays_full_slice)
        update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
      end

      info("## Validation summary")
      for (name, value) in get(opts.eval_metric)
        info(format("{1:>15s} = {2:.4f}", name, value))
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
    _invoke_callbacks(self, opts.callbacks, cb_param, AbstractEpochCallback)
  end # end of all epochs
end

function save_checkpoint(self :: FeedForward, prefix :: AbstractString, param :: CallbackParams)
  save_checkpoint(self.arch, self.arg_params, self.aux_params, prefix, param.curr_epoch)
end
function save_checkpoint(sym :: Symbol, arg_params :: Dict{Base.Symbol, NDArray},
                         aux_params :: Dict{Base.Symbol, NDArray}, prefix :: AbstractString, epoch :: Int)
  save("$prefix-symbol.json", sym)
  save_dict = merge(Dict([symbol("arg:$k") => v for (k,v) in arg_params]),
                    Dict([symbol("aux:$k") => v for (k,v) in aux_params]))
  save_filename = format("{1}-{2:04d}.params", prefix, epoch)
  save(save_filename, save_dict)
  info("Saved checkpoint to '$save_filename'")
end

