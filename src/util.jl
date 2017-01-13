################################################################################
# Dataset related utilities
################################################################################
function get_data_dir()
  data_dir = joinpath(Pkg.dir("MXNet"), "data")
  mkpath(data_dir)
  data_dir
end

function get_mnist_ubyte()
  data_dir  = get_data_dir()
  mnist_dir = joinpath(data_dir, "mnist")
  mkpath(mnist_dir)
  filenames = Dict(:train_data  => "train-images-idx3-ubyte",
                   :train_label => "train-labels-idx1-ubyte",
                   :test_data   => "t10k-images-idx3-ubyte",
                   :test_label  => "t10k-labels-idx1-ubyte")
  filenames = Dict(map((x) -> x[1] => joinpath(mnist_dir, x[2]), filenames))
  if !all(isfile, values(filenames))
    cd(mnist_dir) do
      mnist_dir = download("http://data.dmlc.ml/mxnet/data/mnist.zip", "mnist.zip")
        try
          run(`unzip -u $mnist_dir`)
        catch
          try
            run(pipe(`7z x $mnist_dir`,stdout=DevNull))
          catch
            error("Extraction Failed:No extraction program found in path")
          end
      end
    end
  end
  return filenames
end

function get_cifar10()
  data_dir    = get_data_dir()
  cifar10_dir = joinpath(data_dir, "cifar10")
  mkpath(cifar10_dir)
  filenames = Dict(:train => "cifar/train.rec", :test => "cifar/test.rec")
  filenames = Dict(map((x) -> x[1] => joinpath(cifar10_dir, x[2]), filenames))
  if !all(isfile, values(filenames))
    cd(cifar10_dir) do
      run(`http://data.dmlc.ml/mxnet/data/cifar10.zip`)
        try
          run(`unzip -u cifar10.zip`)
        catch
          try
            run(pipeline(`7z x cifar10.zip`, stdout=DevNull))
          catch
            error("Extraction Failed:No extraction program found in path")
          end
      end
    end
  end

  filenames[:mean] = joinpath(cifar10_dir, "cifar/cifar_mean.bin")
  return filenames
end


################################################################################
# Internal Utilities
################################################################################
function _get_libmx_op_names()
  n = Ref{MX_uint}(0)
  names = Ref{char_pp}(0)

  @mxcall(:MXListAllOpNames, (Ref{MX_uint}, Ref{char_pp}), n, names)

  names = unsafe_wrap(Array, names[], n[])
  return [unsafe_string(x) for x in names]
end
function _get_libmx_op_handle(name :: String)
  handle = Ref{MX_handle}(0)
  @mxcall(:NNGetOpHandle, (char_p, Ref{MX_handle}), name, handle)
  return MX_OpHandle(handle[])
end

# We keep a cache and retrieve the address everytime
# we run Julia, instead of pre-compiling with macro,
# because the actual handle might change in different
# runs
const _libmx_op_cache = Dict{String, MX_OpHandle}()
function _get_cached_libmx_op_handle(name :: String)
  if !haskey(_libmx_op_cache, name)
    handle = _get_libmx_op_handle(name)
    _libmx_op_cache[name] = handle
    return handle
  else
    return _libmx_op_cache[name]
  end
end

function _get_libmx_op_description(name :: String, handle :: MX_OpHandle)
  # get operator information (human readable)
  ref_real_name = Ref{char_p}(0)
  ref_desc = Ref{char_p}(0)
  ref_narg = Ref{MX_uint}(0)

  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  ref_key_narg  = Ref{char_p}(0)
  ref_ret_type  = Ref{char_p}(0)

  @mxcall(:MXSymbolGetAtomicSymbolInfo,
         (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp},
          Ref{char_pp}, Ref{char_pp}, Ref{char_p}, Ref{char_p}),
          handle, ref_real_name, ref_desc, ref_narg, ref_arg_names,
          ref_arg_types, ref_arg_descs, ref_key_narg, ref_ret_type)

  real_name = unsafe_string(ref_real_name[])
  signature = _format_signature(Int(ref_narg[]), ref_arg_names)
  desc = "    " * name * "(" * signature * ")\n\n"
  if real_name != name
    desc *= name * " is an alias of " * real_name * ".\n\n"
  end

  key_narg = unsafe_string(ref_key_narg[])
  if key_narg != ""
    desc *= "**Note**: " * name * " takes variable number of positional inputs. "
    desc *= "So instead of calling as $name([x, y, z], $key_narg=3), "
    desc *= "one should call via $name(x, y, z), and $key_narg will be "
    desc *= "determined automatically.\n\n"
  end

  desc *= unsafe_string(ref_desc[]) * "\n\n"
  desc *= "# Arguments\n"
  desc *= _format_docstring(Int(ref_narg[]), ref_arg_names, ref_arg_types, ref_arg_descs)
  return desc, key_narg
end

function _format_typestring(typestr :: String)
  replace(typestr, r"\bSymbol\b", "SymbolicNode")
end
function _format_docstring(narg::Int, arg_names::Ref{char_pp}, arg_types::Ref{char_pp}, arg_descs::Ref{char_pp}, remove_dup::Bool=true)
  param_keys = Set{String}()

  arg_names  = unsafe_wrap(Array, arg_names[], narg)
  arg_types  = unsafe_wrap(Array, arg_types[], narg)
  arg_descs  = unsafe_wrap(Array, arg_descs[], narg)
  docstrings = String[]

  for i = 1:narg
    arg_name = unsafe_string(arg_names[i])
    if arg_name ∈ param_keys && remove_dup
      continue
    end
    push!(param_keys, arg_name)

    arg_type = _format_typestring(unsafe_string(arg_types[i]))
    arg_desc = unsafe_string(arg_descs[i])
    push!(docstrings, "* `$arg_name::$arg_type`: $arg_desc\n")
  end
  return join(docstrings, "\n")
end

function _format_signature(narg::Int, arg_names::Ref{char_pp})
  arg_names  = unsafe_wrap(Array, arg_names[], narg)

  return join([unsafe_string(name) for name in arg_names] , ", ")
end

