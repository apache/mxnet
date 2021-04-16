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
namespace MxNet.Gluon.RNN
{
    public class ModifierCell : HybridRecurrentCell
    {
        public ModifierCell(RecurrentCell base_cell)
            : base()
        {
            BaseCell = base_cell;
        }

        public RecurrentCell BaseCell { get; }

        public override ParameterDict Params => BaseCell.Params;

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return BaseCell.StateInfo(batch_size);
        }

        public override NDArrayOrSymbolList BeginState(int batch_size = 0, string func = null, FuncArgs args = null)
        {
            BaseCell._modified = false;
            var begin = BaseCell.BeginState(batch_size, func, args);
            BaseCell._modified = true;
            return begin;
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            return default;
        }
    }
}