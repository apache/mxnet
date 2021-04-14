using System;
using System.Collections.Generic;
using System.Text;
using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.Data;
using MxNet.Gluon.Data.Vision.Datasets;
using MxNet.Gluon.Data.Vision.Transforms;
using MxNet.Gluon.Losses;
using MxNet.Gluon.NN;
using MxNet.Initializers;
using MxNet.Optimizers;

namespace BasicExamples
{
    public class CrashCourse_NN
    {
        public static void Run()
        {
            var mnist_train = new FashionMNIST(train: true);
            var (x, y) = mnist_train[0];
            Console.WriteLine($"X shape: {x.shape}, X dtype: {x.dtype}, Y shape: {y.shape}, Y dtype: {y.dtype}");

            var transformer = new Compose(
                                    new ToTensor(),
                                    new Normalize(new MxNet.Tuple<float>(0.13f, 0.31f))
                                    );

            var train = mnist_train.TransformFirst(transformer);
            int batch_size = 256;
            var train_data = new DataLoader(train, batch_size: batch_size, shuffle: true);
            foreach (var (data, label) in train_data)
            {
                Console.WriteLine(data.shape + ", " + label.shape);
                break;
            }

            var mnist_valid = new FashionMNIST(train: false);
            var valid_data = new DataLoader(mnist_valid, batch_size: batch_size, shuffle: true);

            var net = new Sequential();
            net.Add(new Conv2D(channels: 6, kernel_size: (5, 5), activation: ActivationType.Relu),
                    new MaxPool2D(pool_size: (2, 2), strides: (2, 2)),
                    new Conv2D(channels: 16, kernel_size: (3, 3), activation: ActivationType.Relu),
                    new MaxPool2D(pool_size: (2, 2), strides: (2, 2)),
                    new Flatten(),
                    new Dense(120, activation: ActivationType.Relu),
                    new Dense(84, activation: ActivationType.Relu),
                    new Dense(10));

            net.Initialize(new Xavier());

            var softmax_cross_entropy = new SoftmaxCrossEntropyLoss();
            var trainer = new Trainer(net.CollectParams(), new SGD(learning_rate: 0.1f));

            for (int epoch = 0; epoch < 10; epoch++)
            {
                var tic = DateTime.Now;
                float train_loss = 0;
                float train_acc = 0;
                float valid_acc = 0;

                foreach (var (data, label) in train_data)
                {
                    NDArray loss = null;
                    NDArray output = null;
                    // forward + backward
                    using (Autograd.Record())
                    {
                        output = net.Call(data);
                        loss = softmax_cross_entropy.Call(output, label);
                    }

                    loss.Backward();

                    //update parameters
                    trainer.Step(batch_size);

                    //calculate training metrics
                    train_loss += loss.Mean();
                    train_acc += Acc(output, label);
                }

                // calculate validation accuracy
                foreach (var (data, label) in valid_data)
                {
                    valid_acc += Acc(net.Call(data), label);
                }

                Console.WriteLine($"Epoch {epoch}: loss {train_loss / train_data.Length}," +
                                    $" train acc {train_acc / train_data.Length}, " +
                                    $"test acc {train_acc / train_data.Length} " +
                                    $"in {(DateTime.Now - tic).TotalMilliseconds} ms");
            }

            net.SaveParameters("net.params");
        }

        public static float Acc(NDArray output, NDArray label)
        {
            // output: (batch, num_output) float32 ndarray
            // label: (batch) int32 ndarray
            return nd.Equal(output.Argmax(axis: 1), label.AsType(DType.Float32)).Mean();
        }
    }
}
