/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using MxNet.Initializers;

namespace MxNet.Gluon.NN
{
    public class Dense : HybridBlock
    {
        public Dense(int units, ActivationType? activation = null, bool use_bias = true, bool flatten = true,
            DType dtype = null, string weight_initializer = null, string bias_initializer = "zeros",
            int in_units = 0) : base()
        {
            Units = units;
            InUnits = in_units;
            Act = activation != null ? new Activation(activation.Value) : null;
            UseBias = use_bias;
            Flatten_ = flatten;
            DataType = dtype;
            this["weight"] = Params.Get("weight", OpGradReq.Write, new Shape(units, in_units), dtype,
                init: Initializer.Get(weight_initializer), allow_deferred_init: true);

            if (UseBias)
                this["bias"] = Params.Get("bias", OpGradReq.Write, new Shape(units), dtype,
                    init: Initializer.Get(bias_initializer), allow_deferred_init: true);
        }

        public int Units { get; set; }

        public Activation Act { get; set; }

        public bool UseBias { get; set; }

        public bool Flatten_ { get; set; }

        public DType DataType { get; set; }

        public int InUnits { get; set; }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            NDArrayOrSymbol output = null;
            var (x, weight, bias) = args;
            output = F.fully_connected(x, weight, bias, Units, !UseBias, Flatten_);

            if (Act != null)
                output = Act.HybridForward(output);

            return output;
        }

        public override string ToString()
        {
            var shape = Params["weight"].Shape;
            return $"{GetType().Name} ({(shape.Dimension >= 2 && shape[1] > 0 ? shape[1].ToString() : "None")} -> {shape[0]}, {(Act == null ? "linear" : Act.ToString())})";
        }
    }
}