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
    public class Embedding : HybridBlock
    {
        public Embedding(int input_dim, int output_dim, DType dtype = null,
            string weight_initializer = null, bool sparse_grad = false, string prefix = null,
            ParameterDict @params = null) : base()
        {
            Input_Dim = input_dim;
            Output_Dim = output_dim;
            Dtype = dtype;
            Sparse_Grad = sparse_grad;
            this["weight"] = Params.Get("weight", OpGradReq.Write, new Shape(input_dim, output_dim), dtype,
                init: Initializer.Get(weight_initializer), allow_deferred_init: true,
                grad_stype: Sparse_Grad ? StorageStype.RowSparse : StorageStype.Default);
        }

        public int Input_Dim { get; }
        public int Output_Dim { get; }
        public DType Dtype { get; }
        public bool Sparse_Grad { get; }
        public Parameter Weight { get; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            var weight = args[0];

            if (x.IsNDArray)
                return nd.Embedding(x.NdX, weight.NdX, Input_Dim, Output_Dim, Dtype, Sparse_Grad);

            return sym.Embedding(x.SymX, weight.SymX, Input_Dim, Output_Dim, Dtype, Sparse_Grad);
        }

        public override string ToString()
        {
            return $"{GetType().Name}({Input_Dim} -> {Output_Dim}, {Dtype ?? DType.Float32})";
        }

    }
}