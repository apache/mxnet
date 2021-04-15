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
using MxNet.Sym.Numpy;
using System.Linq;

namespace MxNet.Gluon.RNN
{
    public class ResidualCell : ModifierCell
    {
        public ResidualCell(RecurrentCell base_cell) : base(base_cell)
        {
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            var (output, states) = BaseCell.Call(x, args);
            if (x.IsNDArray)
                output = nd.ElemwiseAdd(output, x);
            else
                output = sym.ElemwiseAdd(output, x, $"t{_counter}_fwd");

            return (output, states);
        }

        public override (NDArrayOrSymbol[], NDArrayOrSymbol[]) Unroll(int length, NDArrayOrSymbol[] inputs,
            NDArrayOrSymbol[] begin_state = null, string layout = "NTC", bool? merge_outputs = null,
            _Symbol valid_length = null)
        {
            Reset();
            BaseCell._modified = false;
            var (outputs, states) = BaseCell.Unroll(length, inputs, begin_state, layout, merge_outputs, valid_length);
            BaseCell._modified = true;

            if (!merge_outputs.HasValue) merge_outputs = outputs.Length > 1;

            if (merge_outputs.Value)
                outputs = outputs.Zip(inputs, (i, j) =>
                {
                    if (i.IsNDArray)
                        return new NDArrayOrSymbol(nd.ElemwiseAdd(i, j));

                    return new NDArrayOrSymbol(sym.ElemwiseAdd(i, j));
                }).ToArray();

            return (outputs, states);
        }
    }
}