using System;
using System.Collections.Generic;
using System.Text;
using MxNet;
using MxNet.NN.Constraints;
using MxNet.NN.Initializers;
using MxNet.NN.Regularizers;

namespace MxNet.NN.Layers.Misc
{
    public class Embedding : BaseLayer
    {
        public int InputDim { get; set; }

        public int OutputDim { get; set; }

        public BaseInitializer EmbeddingsInitializer { get; set; }

        public BaseConstraint EmbeddingsConstraint { get; set; }

        public BaseRegularizer EmbeddingsRegularizer { get; set; }

        public Embedding(int inputDim, int outputDim, BaseInitializer embeddingsInitializer=null, BaseRegularizer embeddingsRegularizer=null,BaseConstraint embeddingsConstraint = null)
            :base("embedding")
        {
            InputDim = inputDim;
            OutputDim = outputDim;
            EmbeddingsInitializer = embeddingsInitializer ?? new RandomUniform();
            EmbeddingsConstraint = embeddingsConstraint;
            EmbeddingsRegularizer = embeddingsRegularizer;
        }

        public override Symbol Build(Symbol x)
        {
            var weightName = UUID.GetID(ID + "_w");
            InitParams.Add(weightName, EmbeddingsInitializer);
            ConstraintParams.Add(weightName, EmbeddingsConstraint);
            RegularizerParams.Add(weightName, EmbeddingsRegularizer);
            return sym.Embedding(x, Symbol.Variable(weightName), InputDim, OutputDim, symbol_name: ID);
        }
    }
}
