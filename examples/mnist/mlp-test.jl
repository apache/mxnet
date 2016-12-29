# This file is primarily to be included from runtest.jl. We tried to cover various
# features of MXNet.jl in this example in order to detect regression errors.

module MNISTTest
using MXNet
using Base.Test

function get_mnist_mlp()
  mlp = @mx.chain mx.Variable(:data)             =>
    mx.FullyConnected(name=:fc1, num_hidden=128) =>
    mx.Activation(name=:relu1, act_type=:relu)   =>
    mx.FullyConnected(name=:fc2, num_hidden=64)  =>
    mx.Activation(name=:relu2, act_type=:relu)   =>
    mx.FullyConnected(name=:fc3, num_hidden=10)  =>
    mx.SoftmaxOutput(name=:softmax)
  return mlp
end

function get_mnist_data(batch_size=100)
  include("mnist-data.jl")
  return get_mnist_providers(batch_size)
end

function mnist_fit_and_predict(optimizer, initializer, n_epoch)
  mlp = get_mnist_mlp()
  train_provider, eval_provider = get_mnist_data()

  # setup model
  model = mx.FeedForward(mlp, context=mx.cpu())

  # fit parameters
  cp_prefix = "mnist-test-cp"
  mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=n_epoch,
         initializer=initializer, callbacks=[mx.speedometer(), mx.do_checkpoint(cp_prefix, save_epoch_0=true)])

  # make sure the checkpoints are saved
  @test isfile("$cp_prefix-symbol.json")
  for i_epoch = 0:n_epoch
    @test isfile(mx.format("{1}-{2:04d}.params", cp_prefix, i_epoch))
  end
  mlp_load = mx.load("$cp_prefix-symbol.json", mx.SymbolicNode)
  @test mx.to_json(mlp_load) == mx.to_json(mlp)
  mlp_load = mx.from_json(readall("$cp_prefix-symbol.json"), mx.SymbolicNode)
  @test mx.to_json(mlp_load) == mx.to_json(mlp)

  #--------------------------------------------------------------------------------
  # the predict API
  probs = mx.predict(model, eval_provider)

  # collect all labels from eval data
  labels = Array[]
  for batch in eval_provider
    push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
  end
  labels = cat(1, labels...)

  # Now we use compute the accuracy
  correct = 0
  for i = 1:length(labels)
    # labels are 0...9
    if indmax(probs[:,i]) == labels[i]+1
      correct += 1
    end
  end
  accuracy = 100correct/length(labels)
  println(mx.format("Accuracy on eval set: {1:.2f}%", accuracy))

  # try to call visualization
  dot_code = mx.to_graphviz(mlp)

  return accuracy
end

function test_mnist_mlp()
  @test mnist_fit_and_predict(mx.SGD(lr=0.1, momentum=0.9), mx.UniformInitializer(0.01), 2) > 90
  @test mnist_fit_and_predict(mx.ADAM(), mx.NormalInitializer(), 2) > 90
  @test mnist_fit_and_predict(mx.AdaGrad(), mx.NormalInitializer(), 2) > 90
  @test mnist_fit_and_predict(mx.AdaDelta(), mx.NormalInitializer(), 2) > 90
  @test mnist_fit_and_predict(mx.AdaMax(), mx.NormalInitializer(), 2) > 90
  @test mnist_fit_and_predict(mx.RMSProp(), mx.NormalInitializer(), 2) > 90
  @test mnist_fit_and_predict(mx.Nadam(), mx.NormalInitializer(), 2) > 90
end

test_mnist_mlp()

end # module MNISTTest
