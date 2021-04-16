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

namespace MxNet.Gluon.RNN
{
    public class DropoutCell : HybridRecurrentCell
    {
        private readonly Shape _axes;
        private readonly float _rate;

        public DropoutCell(float rate, Shape axes = null) : base()
        {
            _rate = rate;
            _axes = axes;
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return new StateInfo[0];
        }

        public override string Alias()
        {
            return "dropout";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            if (_rate > 0)
            {
                if (x.IsNDArray)
                    x = nd.Dropout(x, _rate, axes: _axes);
                else
                    x = sym.Dropout(x, _rate, axes: _axes, symbol_name: $"t{_counter}_fwd");
            }
            
            return (x, args);
        }

        public override (NDArrayOrSymbolList, NDArrayOrSymbolList) Unroll(int length, NDArrayOrSymbolList inputs,
            NDArrayOrSymbolList begin_state = null, string layout = "NTC", bool? merge_outputs = null,
            _Symbol valid_length = null)
        {
            return base.Unroll(length, inputs, begin_state, layout);
        }
    }
}