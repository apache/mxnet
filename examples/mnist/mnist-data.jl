function get_mnist_providers(batch_size::Int; data_name=:data, label_name=:softmax_label, flat=true)
  # download MNIST into Pkg.dir("MXNet")/data/mnist if not exist
  filenames = mx.get_mnist_ubyte()

  # data provider
  train_provider = mx.MNISTProvider(image=filenames[:train_data],
                                    label=filenames[:train_label],
                                    data_name=data_name, label_name=label_name,
                                    batch_size=batch_size, shuffle=true, flat=flat, silent=true)
  eval_provider = mx.MNISTProvider(image=filenames[:test_data],
                                   label=filenames[:test_label],
                                   data_name=data_name, label_name=label_name,
                                   batch_size=batch_size, shuffle=false, flat=flat, silent=true)

  return (train_provider, eval_provider)
end
