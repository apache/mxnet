using System;
using System.Collections.Generic;
using System.Text;
using MxNet;
using Newtonsoft.Json;
using MxNet.NN.Initializers;
using MxNet.NN.Layers.Activations;
using MxNet.NN.Regularizers;
using MxNet.NN.Constraints;

namespace MxNet.NN.Layers
{
    public class Dense : BaseLayer
    {
        public int Dim { get; set; }

        public string Activation { get; set; }

        public bool UseBias { get; set; }

        public BaseInitializer KernalInitializer { get; set; }

        public BaseInitializer BiasInitializer { get; set; }

        public BaseConstraint KernalConstraint { get; set; }

        public BaseConstraint BiasConstraint { get; set; }

        public BaseRegularizer KernalRegularizer { get; set; }

        public BaseRegularizer BiasRegularizer { get; set; }

        public Dense(int dim, string activation = ActivationType.Linear, 
                    BaseInitializer kernalInitializer = null, BaseRegularizer kernalRegularizer = null, BaseConstraint kernalConstraint = null,
                    bool useBias = false, BaseInitializer biasInitializer = null, BaseRegularizer biasRegularizer=null, BaseConstraint biasConstraint = null)
            : base("dense")
        {
            Dim = dim;
            Activation = activation;
            UseBias = useBias;
            KernalInitializer = kernalInitializer ?? new GlorotUniform();
            BiasInitializer = biasInitializer ?? new Zeros();
            KernalConstraint = kernalConstraint;
            BiasConstraint = biasConstraint;
            KernalRegularizer = kernalRegularizer;
            BiasRegularizer = biasRegularizer;
        }

        public override Symbol Build(Symbol data)
        {
            var weightName = UUID.GetID(ID + "_w");
            var biasName = UUID.GetID(ID + "_b");

            var bias = UseBias ? Symbol.Variable(biasName) : null;

            InitParams.Add(weightName, KernalInitializer);
            if(UseBias)
                InitParams.Add(biasName, BiasInitializer);

            ConstraintParams.Add(weightName, KernalConstraint);
            if(UseBias)
                ConstraintParams.Add(biasName, BiasConstraint);

            RegularizerParams.Add(weightName, KernalRegularizer);
            if(UseBias)
                RegularizerParams.Add(biasName, BiasRegularizer);

            var l = sym.FullyConnected(data, Symbol.Variable(weightName), Dim, bias, !UseBias, true, ID);
            if (Activation != ActivationType.Linear)
            {
                var act = ActivationRegistry.Get(Activation);
                l = act.Build(l);
            }

            return l;
        }
    }
}
