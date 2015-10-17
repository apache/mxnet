abstract AbstractEstimator

type FeedForward <: AbstractEstimator
  arch        :: Symbol
  ctx         :: Vector{Context}

  arg_params  :: Dict{Base.Symbol, NDArray}
  aux_params  :: Dict{Base.Symbol, NDArray}

  # leave the rest fields undefined
  FeedForward(arch :: Symbol, ctx :: Vector{Context}) = new(arch, ctx)
end

function _check_arguments(symbol :: Symbol)
  arg_names = list_arguments(symbol)
  @assert(length(unique(arg_names)) == length(arg_names), "Duplicated names in arguments $arg_names")
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

function _init_params(self :: FeedForward, data :: AbstractDataProvider, initializer)
  # all arg names, including data, label, and parameters
  arg_names    = list_arguments(self.arch)

  data_shapes  = provide_data(data)
  label_shapes = provide_label(data)
  data_names   = [x[1] for x in data_shapes]
  label_names  = [x[1] for x in label_shapes]

  param_names = setdiff(arg_names, data_names ∪ label_names)
  aux_names   = list_auxiliary_states(self.arch)

  arg_shapes, grad_shapes, aux_shapes = infer_shape(self.arch; data_shapes...)
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

  return (param_names, aux_names)
end

function _create_kvstore(kv_type :: Base.Symbol, num_device :: Int, arg_params :: Dict{Base.Symbol,NDArray})
  if num_device == 1 && !ismatch(r"dist", string(kv_type))
    kv = nothing
  else
    if kv_type == :local
      max_size = maximum([prod(size(param)) for (k,param) in arg_params])
      if max_size < 1024 * 1024 * 16
        kv_type = :loca_update_cpu
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

function fit(self :: FeedForward, optimizer :: AbstractOptimizer, data :: AbstractDataProvider;
             initializer :: AbstractInitializer = UniformInitializer(0.01),
             epoch_stop :: Int = 10, epoch_start :: Int = 1,
             eval_data :: Union{Void, AbstractDataProvider} = nothing,
             eval_metric :: AbstractEvalMetric = Accuracy(),
             kvstore :: Union{Base.Symbol, KVStore} = :local)

  info("Start training on $(self.ctx)")

  batch_size  = get_batch_size(data)
  num_dev     = length(self.ctx)
  slices      = _split_inputs(batch_size, num_dev)

  # initialize parameters
  info("Initializing parameters...")
  param_names, aux_names = _init_params(self, data, initializer)

  # setup kvstore
  if isa(kvstore, Base.Symbol)
    info("Creating KVStore...")
    kvstore, update_on_kvstore = _create_kvstore(kvstore, length(self.ctx), self.arg_params)
  end

  train_execs = Array(Executor, num_dev)
  for i = 1:num_dev
    data_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_data(data)]
    label_shapes = [k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_label(data)]
    train_execs[i] = simple_bind(self.arch, self.ctx[i]; grad_req=GRAD_WRITE, data_shapes..., label_shapes...)
  end

  # set up input data structures
  data_names  = [x[1] for x in provide_data(data)]
  label_names = [x[1] for x in provide_label(data)]

  data_arrays = Vector{NDArray}[[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                                for name in data_names]
  label_arrays = Vector{NDArray}[[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)]
                                 for name in label_names]

  param_arrays = Vector{NDArray}[[exec.arg_arrays[i] for exec in train_execs] for i = 1:length(param_names)]
  grad_arrays  = Vector{NDArray}[[exec.grad_arrays[i] for exec in train_execs] for i = 1:length(param_names)]

  optimizer.inv_batch_size = 1.0/batch_size

  if !update_on_kvstore
    updater = get_updater(self.optimizer)
  end

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

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
  cpu_label_arrays_full_slice = [(1:batch_size, x) for x in label_arrays]

  # now start training...
  for i_epoch = epoch_start:epoch_stop
    time_start = time()
    reset!(eval_metric)
    n_batch = 0

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

      n_batch += 1

      # update evaluation metric on training set
      load_label!(batch, cpu_label_arrays_full_slice)
      update!(eval_metric, cpu_label_arrays, cpu_output_arrays)
    end # end of one epoch

    time_stop = time()
    info("== Epoch {1:0>3d} ==========", i_epoch)
    info("## Training summary")
    for (name, value) in get(eval_metric)
      info("{1>15s} = {2:.4f}", name, value)
    end
    info("{1>15s} = {2:.2f} seconds", "time", (time_stop-time_start)/1e9)

    # evaluation on validation set
    if !isa(eval_data, Void)
      # because we are re-using the memory allocated for the training network,
      # the batch_size of the validation dataset must be the same as the training
      # batch_size
      @assert(get_batch_size(eval_data) == batch_size)

      reset!(eval_metric)
      for batch in eval_data
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
        update!(eval_metric, cpu_label_arrays, cpu_output_arrays)
      end

      info("## Validation summary")
      for (name, value) in get(eval_metric)
        info("{1>15s} = {2:.4f}", name, value)
      end
    end
  end # end of all epochs
end
