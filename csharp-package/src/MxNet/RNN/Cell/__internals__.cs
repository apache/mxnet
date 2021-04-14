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
using MxNet.RecurrentLayer;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using MxNet.Gluon.RNN;

namespace MxNet.RNN.Cell
{
    internal class __internals__
    {
        public static Shape[] CellsStateShape(BaseRNNCell[] cells)
        {
            List<Shape> ret = new List<Shape>();
            foreach (var item in cells)
            {
                ret.AddRange(item.StateShape);
            }

            return ret.ToArray();
        }

        public static StateInfo[] CellsStateInfo(BaseRNNCell[] cells)
        {
            List<StateInfo> ret = new List<StateInfo>();
            foreach (var item in cells)
            {
                ret.AddRange(item.StateInfo);
            }

            return ret.ToArray();
        }

        public static SymbolList CellsBeginState(BaseRNNCell[] cells, string func, FuncArgs kwargs = null)
        {
            SymbolList ret = new SymbolList();
            foreach (var item in cells)
            {
                ret.Add(item.BeginState(func, kwargs));
            }

            return ret.ToArray();
        }

        public static NDArrayDict CellsUnpackWeights(BaseRNNCell[] cells, NDArrayDict args)
        {
            SymbolList ret = new SymbolList();
            foreach (var item in cells)
            {
                args = item.UnpackWeights(args);
            }

            return args;
        }

        public static NDArrayDict CellsPackWeights(BaseRNNCell[] cells, NDArrayDict args)
        {
            SymbolList ret = new SymbolList();
            foreach (var item in cells)
            {
                args = item.PackWeights(args);
            }

            return args;
        }

        public static (Symbol, int) NormalizeSequence(int length, SymbolList inputs, string layout, bool merge, string in_layout= null)
        {
            int axis = layout.ToCharArray().ToList().IndexOf('T');
            int in_axis = !string.IsNullOrWhiteSpace(in_layout) ? in_layout.ToCharArray().ToList().IndexOf('T') : axis;
            if (inputs.Length == 1)
            {
                if (!merge)
                {
                    if (inputs[0].ListOutputs().Count != 1)
                        throw new Exception("unroll doesn't allow grouped symbol as input. Please convert " +
                                                "to list with list(inputs) first or let unroll handle splitting.");

                    inputs = sym.Split(inputs[0], length, in_axis, true);
                }
            }
            else
            {
                if (inputs.Length != length)
                    throw new Exception("input length not matching");

                if(merge)
                {
                    inputs = inputs.Select(i => sym.ExpandDims(i, axis)).ToList();
                    inputs = sym.Concat(inputs, axis);
                    in_axis = axis;
                }
            }

            if (inputs.Length == 1)
                inputs[0] = sym.SwapAxis(inputs[0], axis, in_axis);

            return (inputs, axis);
        }
    }
}
