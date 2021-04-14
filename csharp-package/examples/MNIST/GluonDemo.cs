using System;
using System.Linq;
using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.Losses;
using MxNet.Gluon.NN;
using MxNet.Initializers;
using MxNet.IO;
using MxNet.Gluon.Metrics;
using MxNet.Optimizers;
using NumpyDotNet;

namespace MNIST
{
    public class GluonDemo
    {
        public static void RunSimple()
        {
            var mnist = TestUtils.GetMNIST(); //Get the MNIST dataset, it will download if not found
            var batch_size = 100; //Set training batch size
            var train_data = new NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size);
            var val_data = new NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size);
            
            // Define simple network with dense layers
            var net = new Sequential();
            net.Add(new Dense(128, ActivationType.Relu));
            net.Add(new Dense(64, ActivationType.Relu));
            net.Add(new Dense(10));

            //Set context, multi-gpu supported
            var gpus = TestUtils.ListGpus();
            var ctx = gpus.Count > 0 ? gpus.Select(x => Context.Gpu(x)).ToArray() : new[] { Context.Cpu(0) };

            //Initialize the weights
            net.Initialize(new Xavier(magnitude: 2.24f), ctx);

            //Create the trainer with all the network parameters and set the optimizer
            var trainer = new Trainer(net.CollectParams(), new Adam());

            var epoch = 10;
            var metric = new Accuracy(); //Use Accuracy as the evaluation metric.
            var softmax_cross_entropy_loss = new SoftmaxCrossEntropyLoss();
            float lossVal = 0; //For loss calculation
            for (var iter = 0; iter < epoch; iter++)
            {
                var tic = DateTime.Now;
                // Reset the train data iterator.
                train_data.Reset();
                lossVal = 0;

                // Loop over the train data iterator.
                while (!train_data.End())
                {
                    var batch = train_data.Next();

                    // Splits train data into multiple slices along batch_axis
                    // and copy each slice into a context.
                    var data = Utils.SplitAndLoad(batch.Data[0], ctx, batch_axis: 0);

                    // Splits train labels into multiple slices along batch_axis
                    // and copy each slice into a context.
                    var label = Utils.SplitAndLoad(batch.Label[0], ctx, batch_axis: 0);

                    var outputs = new NDArrayList();

                    // Inside training scope
                    NDArray loss = null;
                    for (int i = 0; i < data.Length; i++)
                    {
                        using (var ag = Autograd.Record())
                        {
                            var x = data[i];
                            var y = label[i];
                            var z = net.Call(x);
                            // Computes softmax cross entropy loss.
                            loss = softmax_cross_entropy_loss.Call(z, y);
                            outputs.Add(z);
                        }

                        // Backpropagate the error for one iteration.
                        loss.Backward();
                        lossVal += loss.Mean();
                    }

                    // Updates internal evaluation
                    metric.Update(label, outputs.ToArray());

                    // Make one step of parameter update. Trainer needs to know the
                    // batch size of data to normalize the gradient by 1/batch_size.
                    trainer.Step(batch.Data[0].shape[0]);
                }

                var toc = DateTime.Now;

                // Gets the evaluation result.
                var (name, acc) = metric.Get();

                // Reset evaluation result to initial state.
                metric.Reset();
                Console.Write($"Loss: {lossVal} ");
                Console.WriteLine($"Training acc at epoch {iter}: {name}={(acc * 100).ToString("0.##")}%, Duration: {(toc - tic).TotalSeconds.ToString("0.#")}s");
            }
        }

        public static void RunConv()
        {
            var mnist = TestUtils.GetMNIST();
            var batch_size = 128;
            var train_data = new NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, true);
            var val_data = new NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size);

            var net = new Sequential();
            net.Add(new Conv2D(20, kernel_size: (5, 5), activation: ActivationType.Tanh));
            net.Add(new MaxPool2D(pool_size: (2, 2), strides: (2, 2)));
            net.Add(new Conv2D(50, kernel_size: (5, 5), activation: ActivationType.Tanh));
            net.Add(new MaxPool2D(pool_size: (2, 2), strides: (2, 2)));
            net.Add(new Flatten());
            net.Add(new Dense(500, ActivationType.Tanh));
            net.Add(new Dense(10));

            var gpus = TestUtils.ListGpus();
            var ctx = gpus.Count > 0 ? gpus.Select(x => Context.Gpu(x)).ToArray() : new[] { Context.Cpu(0) };

            net.Initialize(new Xavier(magnitude: 2.24f), ctx);
            var trainer = new Trainer(net.CollectParams(), new SGD(learning_rate: 0.02f));

            var epoch = 10;
            var metric = new Accuracy();
            var softmax_cross_entropy_loss = new SoftmaxCELoss();
            float lossVal = 0;
            for (var iter = 0; iter < epoch; iter++)
            {
                var tic = DateTime.Now;
                train_data.Reset();
                lossVal = 0;
                while (!train_data.End())
                {
                    var batch = train_data.Next();
                    var data = Utils.SplitAndLoad(batch.Data[0], ctx, batch_axis: 0);
                    var label = Utils.SplitAndLoad(batch.Label[0], ctx, batch_axis: 0);

                    var outputs = new NDArrayList();
                    using (var ag = Autograd.Record())
                    {
                        for (var i = 0; i < data.Length; i++)
                        {
                            var x = data[i];
                            var y = label[i];

                            var z = net.Call(x);
                            NDArray loss = softmax_cross_entropy_loss.Call(z, y);
                            loss.Backward();
                            lossVal += loss.Mean();
                            outputs.Add(z);
                        }

                        //outputs = Enumerable.Zip(data, label, (x, y) =>
                        //{
                        //    var z = net.Call(x);
                        //    NDArray loss = softmax_cross_entropy_loss.Call(z, y);
                        //    loss.Backward();
                        //    lossVal += loss.Mean();
                        //    return z;
                        //}).ToList();
                    }

                    metric.Update(label, outputs.ToArray());
                    trainer.Step(batch.Data[0].shape[0]);
                }

                var toc = DateTime.Now;

                var (name, acc) = metric.Get();
                metric.Reset();
                Console.Write($"Loss: {lossVal} ");
                Console.WriteLine($"Training acc at epoch {iter}: {name}={(acc * 100).ToString("0.##")}%, Duration: {(toc - tic).TotalSeconds.ToString("0.#")}s");
            }
        }
    }
}