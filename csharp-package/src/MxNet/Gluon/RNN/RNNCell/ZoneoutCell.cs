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
using System.Linq;

namespace MxNet.Gluon.RNN
{
    public class ZoneoutCell : ModifierCell
    {
        private NDArrayOrSymbol _prev_output;

        public ZoneoutCell(RecurrentCell base_cell, float zoneout_outputs = 0, float zoneout_states = 0) :
            base(base_cell)
        {
            ZoneoutOutputs = zoneout_outputs;
            ZoneoutStates = zoneout_states;
            _prev_output = null;
        }

        public float ZoneoutOutputs { get; }
        public float ZoneoutStates { get; }

        public override string Alias()
        {
            return "zoneout";
        }

        public override void Reset()
        {
            base.Reset();
            _prev_output = null;
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            var (cell, p_outputs, p_states) = (BaseCell, ZoneoutOutputs, ZoneoutStates);
            var (next_output, next_states) = cell.Call(x, args);

            NDArrayOrSymbol mask(float p, NDArrayOrSymbol like)
            {
                if (x.IsNDArray) return nd.Dropout(nd.OnesLike(x), p);

                return sym.Dropout(sym.OnesLike(x), p);
            }

            var prev_output = _prev_output;
            if (prev_output == null)
                prev_output = x.IsNDArray
                    ? new NDArrayOrSymbol(nd.ZerosLike(next_output))
                    : new NDArrayOrSymbol(sym.ZerosLike(next_output));

            NDArrayOrSymbol output = null;
            NDArrayOrSymbol[] states = null;
            if (x.IsNDArray)
            {
                output = p_outputs != 0
                    ? new NDArrayOrSymbol(nd.Where(mask(p_outputs, next_output), next_output, prev_output))
                    : next_output;

                if (p_states == 0)
                    states = next_states;
                else
                    next_states.Zip(states,
                        (new_s, old_s) =>
                        {
                            return new NDArrayOrSymbol(nd.Where(mask(p_states, new_s), new_s, old_s));
                        }).ToArray();
            }
            else if (x.IsSymbol)
            {
                output = p_outputs != 0
                    ? new NDArrayOrSymbol(sym.Where(mask(p_outputs, next_output), next_output, prev_output))
                    : next_output;

                if (p_states == 0)
                    states = next_states;
                else
                    next_states.Zip(states,
                        (new_s, old_s) =>
                        {
                            return new NDArrayOrSymbol(sym.Where(mask(p_states, new_s), new_s, old_s));
                        }).ToArray();
            }

            _prev_output = output;

            return (output, states);
        }
    }
}