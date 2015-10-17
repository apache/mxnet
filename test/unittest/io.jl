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
  data_spec = mx.provide_data(mnist_provider)
  label_spec = mx.provide_label(mnist_provider)
  @test data_spec == [(:data, (28,28,1,batch_size))]
  @test label_spec == [(:softmax_label, (batch_size,))]

  n_batch = 0
  for batch in mnist_provider
    if n_batch == 0
      data_array  = mx.empty(28,28,1,batch_size)
      label_array = mx.empty(batch_size)
      # have to use "for i=1:1" to get over the legacy "feature" of using
      # [ ] to do concatenation in Julia
      data_targets = [[(1:batch_size, data_array)] for i = 1:1]
      label_targets = [[(1:batch_size, label_array)] for i = 1:1]

      mx.load_data!(batch, data_targets)
      mx.load_label!(batch, label_targets)

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
