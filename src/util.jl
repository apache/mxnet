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
  filenames = Dict([k => joinpath(mnist_dir, v) for (k,v) in filenames])
  if !all(isfile, values(filenames))
    cd(mnist_dir) do
      mnist_dir = download("http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip", "mnist.zip")
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
  filenames = Dict([k => joinpath(cifar10_dir, v) for (k,v) in filenames])
  if !all(isfile, values(filenames))
    cd(cifar10_dir) do
      run(`wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip`)
        try
          run(`unzip -u cifar10.zip`)
        catch
          try
            run(pipe(`7z x cifar10.zip`,stdout=DevNull))
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
const DOC_EMBED_ANCHOR = "**autogen:EMBED:{1}:EMBED:autogen**"
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

