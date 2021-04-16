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
using System;
using System.Collections.Generic;
using System.Linq;

namespace MxNet.Gluon.RNN
{
    public class HybridSequentialRNNCell : HybridRecurrentCell
    {
        public List<RecurrentCell> _layers;

        public HybridSequentialRNNCell() : base()
        {
            _layers = new List<RecurrentCell>();
        }

        public new SequentialRNNCell this[string i] => (SequentialRNNCell) _childrens[i];

        public void Add(RecurrentCell cell)
        {
            _layers.Add(cell);
            RegisterChild(cell);
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return RNNCell.CellsStateInfo(_childrens.Values.ToArray(), batch_size);
        }

        public override NDArrayOrSymbolList BeginState(int batch_size = 0, string func = null, FuncArgs args = null)
        {
            return RNNCell.CellsBeginState(_childrens.Values.ToArray(), batch_size, func);
        }

        public override NDArrayOrSymbolList Call(NDArrayOrSymbolList args)
        {
            _counter++;
            var inputs = args[0];
            NDArrayOrSymbolList states = args[1].List;
            var next_states = new NDArrayOrSymbolList();
            var p = 0;
            foreach (var cell in _childrens.Values)
            {
                if (cell.GetType().Name == "BidirectionalCell")
                    throw new Exception("BidirectionalCell is not allowed.");

                var n = cell.StateInfo().Length;
                NDArrayOrSymbolList state = new NDArrayOrSymbolList(states.Skip(p).Take(n).ToArray());
                p += n;
                (inputs, state) = cell.Call((inputs, state));
                next_states.Add(state);
            }

            return (inputs, next_states);
        }

        public override (NDArrayOrSymbolList, NDArrayOrSymbolList) Unroll(int length, NDArrayOrSymbolList inputs,
            NDArrayOrSymbolList begin_state = null, string layout = "NTC", bool? merge_outputs = null,
            _Symbol valid_length = null)
        {
            Reset();
            var (inputs1, _, batch_size) = RNNCell.FormatSequence(length, inputs, layout, false);
            inputs = inputs1;
            var num_cells = _childrens.Count;
            begin_state = RNNCell.GetBeginState(this, begin_state, inputs, batch_size);
            var p = 0;
            NDArrayOrSymbolList states = null;

            var next_states = new NDArrayOrSymbolList();
            foreach (var item in _childrens)
            {
                var i = Convert.ToInt32(item.Key);
                var cell = item.Value;
                var n = cell.StateInfo().Length;
                p += n;
                (inputs, states) = cell.Unroll(length, inputs, states, layout, i < num_cells - 1 ? null : merge_outputs,
                    valid_length);
                next_states.Add(states);
            }

            return (inputs, next_states);
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            return default;
        }
    }
}