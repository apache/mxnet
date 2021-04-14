
using CsvHelper;
using MxNet;
using MxNet.IO;
using MxNet.Metrics;
using MxNet.NN;
using MxNet.NN.Data;
using MxNet.NN.Layers;
using MxNet.Optimizers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BostonHousingRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            //Environment.SetEnvironmentVariable("MXNET_ENGINE_TYPE", "NaiveEngine");
            mx.SetDevice(DeviceType.CPU);
            //uint batchSize = 200;
            //uint numFeatures = 13;
            //var x = Symbol.Variable("x");

            //var trainData = ReadCsv("./data/train.csv");
            //var x_train = trainData.SliceAxis(1, 1, 14);
            //var y_train = trainData.SliceAxis(1, 14, 15);

            //NDArrayIter dataIter = new NDArrayIter(x_train, y_train);

            //var fc1 = sym.FullyConnected(x, Symbol.Variable("fc1_w"), null, 64, no_bias: true, symbol_name: "fc1");
            //var fc2 = sym.Relu(sym.FullyConnected(fc1, Symbol.Variable("fc2_w"), null, 32, no_bias: true, symbol_name: "fc2"), "relu2");
            //var fc3 = sym.FullyConnected(fc2, Symbol.Variable("fc3_w"), null, 1, no_bias: true, symbol_name: "fc3");
            //var output = sym.LinearRegressionOutput(fc3, Symbol.Variable("label"), symbol_name: "model");

            //NDArrayDict parameters = new NDArrayDict();
            //parameters["x"] = new NDArray(new Shape(batchSize, numFeatures));
            //parameters["label"] = new NDArray(new Shape(batchSize));
            //output.InferArgsMap(MXNet.Device, parameters, parameters);

            //foreach (var item in parameters.ToList())
            //{
            //    if (item.Key == "x" || item.Key == "label")
            //        continue;

            //    item.Value.SampleUniform();
            //}

            //var opt = new Adam();
            //BaseMetric metric = new MAE();
            //using (var exec = output.SimpleBind(MXNet.Device, parameters))
            //{
            //    dataIter.SetBatch(batchSize);
            //    var argNames = output.ListArguments();
            //    DataBatch batch;
            //    for (int iter = 1; iter <= 1000; iter++)
            //    {
            //        dataIter.Reset();
            //        metric.Reset();

            //        while (dataIter.Next())
            //        {
            //            batch = dataIter.GetDataBatch();
            //            batch.Data.CopyTo(parameters["x"]);
            //            batch.Label.CopyTo(parameters["label"]);
            //            exec.Forward(true);
            //            exec.Backward();

            //            for (var i = 0; i < argNames.Count; ++i)
            //            {
            //                if (argNames[i] == "x" || argNames[i] == "label")
            //                    continue;

            //                opt.Update(iter, i, exec.ArgmentArrays[i], exec.GradientArrays[i]);
            //            }

            //            metric.Update(parameters["label"], exec.Output);
            //        }

            //        Console.WriteLine("Iteration: {0}, Metric: {1}", iter, metric.Get());
            //    }
            //}

            //Global.Device = Context.Cpu();

            ////Read Data
            CsvDataFrame trainReader = new CsvDataFrame("./data/train.csv", true);
            trainReader.ReadCsv();
            var trainX = trainReader[1, 14];
            var trainY = trainReader[14, 15];

            CsvDataFrame valReader = new CsvDataFrame("./data/test.csv", true);
            valReader.ReadCsv();

            var valX = valReader[1, 14];

            NDArrayIter train = new NDArrayIter(trainX, trainY);

            //Build Model
            var model = new Module(13);
            model.Add(new Dense(64, ActivationType.ReLU));
            model.Add(new Dense(32, ActivationType.ReLU));
            model.Add(new Dense(1));

            model.Compile(OptimizerRegistry.Adam(), LossType.MeanSquaredError, new MSE());
            model.Fit(train, 1000, 32);

            Console.ReadLine();
        }

        private static NDArray ReadCsv(string path)
        {
            List<float> data = new List<float>();
            uint cols = 0;
            uint rows = 0;
            using (TextReader fileReader = File.OpenText(path))
            {
                var csv = new CsvReader(fileReader);
                csv.Read();
                
                while (csv.Read())
                {
                    string[] rowData = csv.Parser.Context.Record;
                    cols = (uint)rowData.Length;
                    foreach (string item in rowData)
                    {
                        data.Add(float.Parse(item));
                    }

                    rows++;
                }
            }

            return new NDArray(data.ToArray(), new Shape(rows, cols));
        }
    }
}
