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
  filenames = [k => joinpath(mnist_dir, v) for (k,v) in filenames]
  if !all(isfile, values(filenames))
    cd(mnist_dir) do
      run(`wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip`)
      run(`unzip -u mnist.zip`)
    end
  end
  return filenames
end

function get_cifar10()
  data_dir    = get_data_dir()
  cifar10_dir = joinpath(data_dir, "cifar10")
  mkpath(cifar10_dir)
  filenames = Dict(:train => "cifar/train.rec", :test => "cifar/test.rec")
  filenames = [k => joinpath(cifar10_dir, v) for (k,v) in filenames]
  if !all(isfile, values(filenames))
    cd(cifar10_dir) do
      run(`wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip`)
      run(`unzip -u cifar10.zip`)
    end
  end

  filenames[:mean] = joinpath(cifar10_dir, "cifar/cifar_mean.bin")
  return filenames
end
