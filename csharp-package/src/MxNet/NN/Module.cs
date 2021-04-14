using MxNet;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using MxNet.Optimizers;
using MxNet.Metrics;
using MxNet.NN.Initializers;
using MxNet.NN.Layers;
using MxNet.EventArgs;

namespace MxNet.NN
{
    public partial class Module
    {
        /// <summary>
        ///     Occurs when [on batch end].
        /// </summary>
        public event EventHandler<BatchEndEventArgs> BatchEnd;

        /// <summary>
        ///     Occurs when [on batch start].
        /// </summary>
        public event EventHandler<BatchStartEventArgs> BatchStart;

        /// <summary>
        ///     Occurs when [on epoch end].
        /// </summary>
        public event EventHandler<EpochEndEventArgs> EpochEnd;

        /// <summary>
        ///     Occurs when [on epoch start].
        /// </summary>
        public event EventHandler<EpochStartEventArgs> EpochStart;

        /// <summary>
        ///     Occurs when [on training end].
        /// </summary>
        public event EventHandler<TrainingEndEventArgs> TrainingEnd;

        public uint[] InputShape { get; set; }

        [JsonIgnore]
        public Symbol Model = null;

        public List<BaseLayer> Layers
        {
            get; set;
        }

        [JsonIgnore]
        public Optimizer ModelOptimizer
        {
            get; set;
        }

        [JsonIgnore]
        public EvalMetric Metric { get; set; }

        [JsonIgnore]
        public Symbol TrainingLoss { get; set; }

        [JsonIgnore]
        public EvalMetric TrainMetric { get; set; }

        private Dictionary<string, BaseInitializer> ParamInitializers = new Dictionary<string, BaseInitializer>();

        private NDArrayDict args;

        public Module(params uint[] inputShape)
        {
            Layers = new List<BaseLayer>();
            args = new NDArrayDict();
            SetInput(inputShape);
        }

        public void SetInput(params uint[] inputShape)
        {
            InputShape = inputShape;
        }

        public void Add(BaseLayer l)
        {
            Layers.Add(l);
        }

        public void Compile(Optimizer optimizer, LossType loss, EvalMetric metric = null)
        {
            Metric = metric;
            TrainMetric = metric;
            Model = new Symbol(IntPtr.Zero);
            Model = Symbol.Variable("X");
            ModelOptimizer = optimizer;

            foreach (var layer in Layers)
            {
                Model = layer.Build(Model);
                foreach (var item in ((BaseLayer)layer).InitParams)
                {
                    ParamInitializers.Add(item.Key, item.Value);
                }
            }

            Model = LossRegistry.Get(loss, Model, Symbol.Variable("label"));
        }

        public void Compile(Symbol model, Optimizer optimizer, EvalMetric metric)
        {
            Metric = metric;
            TrainMetric = metric;
            ModelOptimizer = optimizer;
            Model = new Symbol(model.NativePtr);
        }

        public void Compile(OptimizerType optimizer, LossType loss, EvalMetric metric)
        {
            Optimizer opt = OptimizerRegistry.Get(optimizer);
            Compile(opt, loss, metric);
        }

        public void Compile(Symbol model, OptimizerType optimizer, EvalMetric metric)
        {
            Optimizer opt = OptimizerRegistry.Get(optimizer);
            Compile(model, opt, metric);
        }

        public void SaveModel(string folder, bool saveSymbol = true, string moduleFileName = "module", string symbolFileName = "symbol")
        {
            string sequential = JsonConvert.SerializeObject(this, Formatting.Indented,
                                            new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });
            string modulePath = string.Format("{0}/{1}.json", folder, moduleFileName);
            string symbolPath = string.Format("{0}/{1}.json", folder, symbolFileName);
            File.WriteAllText(modulePath, sequential);
            
            if (saveSymbol)
            {
                Model.Save(symbolPath);
            }
        }

        public static Module LoadModel(string folder, bool loadSymbol = true, string moduleFileName = "module", string symbolFileName = "symbol")
        {
            string modulePath = string.Format("{0}/{1}.json", folder, moduleFileName);
            string symbolPath = string.Format("{0}/{1}.json", folder, symbolFileName);

            string seq_json = File.ReadAllText(modulePath);
            Module model = JsonConvert.DeserializeObject<Module>(seq_json,   
                                                new JsonSerializerSettings() { TypeNameHandling = TypeNameHandling.Auto });

            if (loadSymbol && File.Exists(symbolPath))
            {
                model.Model = Symbol.Load(symbolPath);
            }

            return model;
        }

        public void SaveCheckpoint(string folder, int iter = 0)
        {
            string paramFilePath = string.Format("{0}/chkpt_{1}.params", folder, iter);
            NDArray.Save(paramFilePath, args);
        }

        public void LoadCheckpoint(string folder, int iter = 0)
        {
            string paramFilePath = string.Format("{0}/chkpt_{1}.params", folder, iter);
            args = NDArray.LoadToMap(paramFilePath);
        }
    }
}
