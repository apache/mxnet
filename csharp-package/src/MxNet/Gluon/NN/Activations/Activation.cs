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
using System;

namespace MxNet.Gluon.NN
{
    public class Activation : HybridBlock
    {
        public Activation(ActivationType activation) : base()
        {
            ActType = activation;
        }

        public ActivationType ActType { get; set; }

        public override string Alias()
        {
            return Enum.GetName(typeof(ActivationType), ActType);
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            var x = args[0];
            if (x.IsNDArray)
                return nd.Activation(x.NdX, ActType);

            return sym.Activation(x.SymX, ActType, "fwd");
        }

        public override string ToString()
        {
            return $"{GetType().Name}({ActType})";
        }
    }
}