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
