using MxNet;
using System;
using System.Collections.Generic;
using System.Text;
using MxNet.NN.Initializers;
using MxNet.NN.Regularizers;
using MxNet.NN.Constraints;
using MxNet.NN.Layers.Activations;

namespace MxNet.NN.Layers
{
    public class Conv1D : BaseLayer
    {
        public uint Filters { get; set; }

        public uint KernalSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public uint DialationRate { get; set; }

        public string Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Conv1D(uint filters, uint kernalSize, uint strides = 1, uint? padding=null, 
                        uint dialationRate = 1, string activation = ActivationType.Linear, BaseInitializer kernalInitializer = null,
                        BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null, bool useBias = true, 
                        BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer = null, BaseConstraint biasConstraint = null)
            :base("conv1d")
        {
            Filters = filters;
            KernalSize = kernalSize;
            Strides = strides;
            Padding = padding;
            DialationRate = dialationRate;
            Activation = activation;
            UseBias = useBias;
            KernalInitializer = kernalInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
            KernalConstraint = kernalConstraint;
            BiasConstraint = biasConstraint;
            KernalRegularizer = kernalRegularizer;
            BiasRegularizer = biasRegularizer;
        }

        public override Symbol Build(Symbol x)
        {
            var biasName = UUID.GetID(ID + "_b");
            var weightName = UUID.GetID(ID + "_w");
            var bias = UseBias ? Symbol.Variable(biasName) : null;
            Shape pad = null;
            if(Padding.HasValue)
            {
                pad = new Shape(Padding.Value);
            }
            else
            {
                pad = new Shape();
            }

            if (UseBias)
                InitParams.Add(biasName, BiasInitializer);
            InitParams.Add(weightName, KernalInitializer);

            ConstraintParams.Add(weightName, KernalConstraint);
            if (UseBias)
                ConstraintParams.Add(biasName, BiasConstraint);

            RegularizerParams.Add(weightName, KernalRegularizer);
            if (UseBias)
                RegularizerParams.Add(biasName, BiasRegularizer);

            var conv = sym.Convolution(x, Symbol.Variable(weightName), new Shape(KernalSize), Filters, new Shape(Strides),
                                            new Shape(DialationRate), pad, bias, !UseBias, 1, 1024, ConvolutionCudnnTune.Off, false, null, ID);

            if (Activation != ActivationType.Linear)
            {
                var act = ActivationRegistry.Get(Activation);
                conv = act.Build(conv);
            }

            return conv;
        }
    }
}
