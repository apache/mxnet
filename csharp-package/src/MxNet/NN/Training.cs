using MxNet;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Linq;
using MxNet.NN.Initializers;
using System.IO;
using MxNet.NN.Data;
using MxNet.IO;

namespace MxNet.NN
{
    public partial class Module
    {
        private BaseInitializer defaultInitializer = new Initializers.GlorotUniform();

        public void SetDefaultInitializer(BaseInitializer instance)
        {
            defaultInitializer = instance;
        }

        public void Fit(DataIter train, uint epochs = 1, uint batchSize = 32, DataIter validation = null, bool shuffle = false)
        {
            string labelName = "label";

            var label = Symbol.Variable(labelName);

            List<uint> inputShape = new List<uint>();
            inputShape.Add(batchSize);
            inputShape.AddRange(InputShape);

            args["X"] = new NDArray(new Shape(inputShape.ToArray()));
            args[labelName] = new NDArray(new Shape(batchSize));

            Model.InferArgsMap(mx.Device, args, args);

            var defaultInitializer = new Initializers.GlorotUniform();

            foreach (var arg in args)
            {
                if (ParamInitializers.ContainsKey(arg.Key))
                {
                    ParamInitializers[arg.Key].Generate(arg.Value);
                }
                else
                {
                    defaultInitializer.Generate(arg.Value);
                }
            }

            using (var exec = Model.SimpleBind(mx.Device, args))
            {
                var argNames = Model.ListArguments();

                // Start training
                var sw = new Stopwatch();
                for (var iter = 1; iter <= epochs; iter++)
                {
                    uint samples = 0;
                    train.BatchSize = batchSize;
                    train.Reset();
                    Metric.Reset();
                    TrainMetric.Reset();
                    sw.Restart();

                    while (train.IterNext())
                    {
                        samples += batchSize;
                        var dataBatch = train.Next();

                        // Set data and label
                        dataBatch.Data[0].CopyTo(args["X"]);
                        dataBatch.Label[0].CopyTo(args[labelName]);
                        
                        // Compute gradients
                        exec.Forward(true);
                        exec.Backward();
                        TrainMetric.Update(args[labelName], exec.Output);
                        
                        // Update parameters
                        for (var i = 0; i < argNames.Count; ++i)
                        {
                            if (argNames[i] == "X" || argNames[i] == labelName)
                                continue;

                            ModelOptimizer.Update(i, exec.ArgmentArrays[i], exec.GradientArrays[i], null);
                        }
                    }

                    sw.Stop();

                    if (validation != null)
                    {
                        validation.BatchSize = batchSize;
                        validation.Reset();
                        while (validation.IterNext())
                        {
                            var dataBatch = validation.Next();
                            dataBatch.Data[0].CopyTo(args["X"]);
                            dataBatch.Label[0].CopyTo(args[labelName]);
                            NDArray.WaitAll();
                            // Forward pass is enough as no gradient is needed when evaluating
                            exec.Forward(false);
                            Metric.Update(args[labelName], exec.Output);
                        }
                    }

                    var duration = sw.ElapsedMilliseconds == 0 ? 1 : sw.ElapsedMilliseconds;
                    if (validation == null)
                    {
                        Logging.LG($"Epoch: {iter} {Convert.ToInt32(samples * 1000 / duration)} samples/sec Train_Metric: {TrainMetric.Get()}");
                    }
                    else
                    {
                        Logging.LG($"Epoch: {iter} {Convert.ToInt32(samples * 1000 / duration)} samples/sec, Train_Metric: {TrainMetric.Get()}, Val_Metric: {Metric.Get()}");
                    }
                }
            }

            //MXNet.MXNotifyShutdown();
        }

        public NDArray Predict(NDArray x, uint? batchSize = null)
        {
            NDArray result = new NDArray();
            List<float> preds = new List<float>();
            NDArrayIter dataIter = new NDArrayIter(new NDArray[] { x }, null);

            if(!batchSize.HasValue)
                batchSize = x.Shape[0];

            List<uint> inputShape = new List<uint>();
            NDArrayDict predictArgs = new NDArrayDict();
            
            Model.InferArgsMap(mx.Device, predictArgs, args);
            predictArgs["X"] = new NDArray(x.Shape);
            predictArgs["label"] = new NDArray(new Shape(batchSize.Value));
            using (var exec = Model.SimpleBind(mx.Device, predictArgs))
            {
                dataIter.BatchSize = batchSize.Value;
                dataIter.Reset();
                while (dataIter.IterNext())
                {
                    var batch = dataIter.Next();
                    batch.Data[0].CopyTo(predictArgs["X"]);
                    exec.Forward(false);
                    preds.AddRange(exec.Output.GetValues<float>());
                }
            }

            return new NDArray(preds.ToArray()).Reshape((int)x.Shape[0], -1);
        }

    }
}
