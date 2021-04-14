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
    public class Conv2DTranspose : BaseLayer
    {
        public uint Filters { get; set; }

        public Tuple<uint, uint> KernalSize { get; set; }

        public Tuple<uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public string Activation { get; set; }

        public Tuple<uint, uint> DialationRate { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Conv2DTranspose(uint filters, Tuple<uint, uint> kernalSize, Tuple<uint, uint> strides = null, uint? padding = null,
                                Tuple<uint, uint> dialationRate = null,  string activation = ActivationType.Linear, BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null,
                                BaseConstraint kernalConstraint = null, bool useBias = true, BaseInitializer biasInitializer =null, BaseRegularizer biasRegularizer = null,
                                BaseConstraint biasConstraint = null)
            : base("conv2dtranspose")
        {
            Filters = filters;
            KernalSize = kernalSize ?? Tuple.Create<uint, uint>(1, 1);
            Strides = strides ?? Tuple.Create<uint, uint>(1, 1);
            Padding = padding;
            DialationRate = dialationRate ?? Tuple.Create<uint, uint>(1, 1);
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
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value, Padding.Value);
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

            var conv = sym.Deconvolution(x, Symbol.Variable(weightName), new Shape(KernalSize.Item1, KernalSize.Item2),
                                    Filters, new Shape(Strides.Item1, Strides.Item2), new Shape(DialationRate.Item1, DialationRate.Item2), pad,
                                    new Shape(), new Shape(), bias, !UseBias, 1, 512, DeconvolutionCudnnTune.Off, false, null, ID);

            if (Activation != ActivationType.Linear)
            {
                var act = ActivationRegistry.Get(Activation);
                conv = act.Build(conv);
            }

            return conv;
        }
    }
}
