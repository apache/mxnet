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
using MxNet.Numpy;

namespace MxNet.Gluon.NN
{
    public class PReLU : HybridBlock
    {
        private ndarray alpha;

        public PReLU(Initializer alpha_initializer = null) : base(
            )
        {
            AlphaInitializer = alpha_initializer ?? new Initializers.Constant(0.25f);
        }

        public Initializer AlphaInitializer { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            if (x.IsNDArray)
                return nd.LeakyReLU(x.NdX, alpha, ReluActType.Prelu);

            return sym.LeakyReLU(x.SymX, alpha, ReluActType.Prelu, symbol_name: "fwd");
        }

        public override string ToString()
        {
            return GetType().Name;
        }
    }
}