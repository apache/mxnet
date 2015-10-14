module TestIO
using MXNet
using Base.Test

function test_mnist()
  info("IO::MNIST")
  filenames = mx.get_mnist_ubyte()

  batch_size = 10
  mnist_provider = mx.MNISTProvider(image=filenames[:train_data],
                                    label=filenames[:train_label],
                                    batch_size=batch_size, silent=true, shuffle=false)
  spec = mx.provides(mnist_provider)
  spec = Dict(spec)
  @test haskey(spec, :data)
  @test haskey(spec, :softmax_label)
  @test spec[:data] == (28,28,1,batch_size)
  @test spec[:softmax_label] == (batch_size,)

  n_batch = 0
  for batch in mnist_provider
    if n_batch == 0
      data_array  = mx.empty(28,28,1,batch_size)
      label_array = mx.empty(batch_size)
      targets = Dict(:data => [(1:batch_size, data_array)],
                     :softmax_label => [(1:batch_size, label_array)])

      mx.load_data!(batch, targets)

      true_labels = [5,0,4,1,9,2,1,3,1,4] # the first 10 labels in MNIST train
      got_labels  = Int[copy(label_array)...]
      @test true_labels == got_labels
    end

    n_batch += 1
  end

  @test n_batch == 60000 / batch_size
end

test_mnist()

end
