using System;
using System.Linq;
using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.NN;
using MxNet.Initializers;
using MxNet.IO;
using MxNet.Metrics;
using MxNet.Optimizers;

namespace ORGate
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var trainX = new NDArray(new float[] {0, 0, 0, 1, 1, 0, 1, 1}).Reshape(4, 2);
            var trainY = new NDArray(new float[] {0, 1, 1, 0});

            var batch_size = 2;
            var train_data = new NDArrayIter(trainX, trainY, batch_size);
            var val_data = new NDArrayIter(trainX, trainY, batch_size);

            var net = new Sequential();
            net.Add(new Dense(64, ActivationType.Relu));
            net.Add(new Dense(1));

            var gpus = TestUtils.ListGpus();
            var ctxList = gpus.Count > 0 ? gpus.Select(x => Context.Gpu(x)).ToArray() : new[] {Context.Cpu()};

            net.Initialize(new Uniform(), ctxList.ToArray());
            var trainer = new Trainer(net.CollectParams(), new Adam());
            var epoch = 1000;
            var metric = new BinaryAccuracy();
            var binary_crossentropy = new LogisticLoss();
            float lossVal = 0;
            for (var iter = 0; iter < epoch; iter++)
            {
                train_data.Reset();
                lossVal = 0;
                while (!train_data.End())
                {
                    var batch = train_data.Next();
                    var data = Utils.SplitAndLoad(batch.Data[0], ctxList);
                    var label = Utils.SplitAndLoad(batch.Label[0], ctxList);
                    NDArrayList outputs = null;
                    using (var ag = Autograd.Record())
                    {
                        outputs = Enumerable.Zip(data, label, (x, y) =>
                        {
                            var z = net.Call(x);
                            NDArray loss = binary_crossentropy.Call(z, y);
                            loss.Backward();
                            lossVal += loss.Mean();
                            return z;
                        }).ToList();
                    }

                    metric.Update(label, outputs.ToArray());
                    trainer.Step(batch.Data[0].Shape[0]);
                }

                var (name, acc) = metric.Get();
                metric.Reset();
                Console.WriteLine($"Loss: {lossVal}");
                Console.WriteLine($"Training acc at epoch {iter}: {name}={acc * 100}%");
            }
        }
    }
}