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
using MxNet.Gluon.RNN;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Reflection;
using MxNet.RNN.Cell;

namespace MxNet.RecurrentLayer
{
    public abstract class BaseRNNCell
    {
        internal bool _own_params;
        internal string _prefix;
        internal RNNParams _params;
        internal bool _modified;
        internal int _init_counter;
        internal int _counter;
        internal int _num_hidden;
        internal List<BaseRNNCell> _cells = new List<BaseRNNCell>();

        public virtual RNNParams Params
        {
            get
            {
                _own_params = false;
                return _params;
            }
        }

        public abstract StateInfo[] StateInfo { get; }

        public virtual Shape[] StateShape
        {
            get
            {
                return StateInfo.Select(x => (x.Shape)).ToArray();
            }
        }

        public virtual string[] GateNames
        {
            get
            {
                return new string[] { "" };
            }
        }

        public BaseRNNCell(string prefix, RNNParams @params = null)
        {
            if (@params == null)
            {
                @params = new RNNParams(prefix);
                _own_params = true;
            }
            else
                _own_params = false;

            _prefix = prefix;
            _params = @params;
            _modified = false;

            Reset();
        }

        public virtual void Reset()
        {
            _init_counter = -1;
            _counter = -1;
            foreach (var cell in _cells)
            {
                cell.Reset();
            }
        }

        public abstract (Symbol, SymbolList) Call(Symbol inputs, SymbolList states);

        public virtual SymbolList BeginState(string func = "sym.Zeros", FuncArgs kwargs = null)
        {
            if (_modified)
                throw new Exception("After applying modifier cells (e.g. DropoutCell) the base "  +
                                        "cell cannot be called directly. Call the modifier cell instead.");

            SymbolList states = new SymbolList();
            for (int i = 0; i < StateInfo.Length; i++)
            {
                var info = StateInfo[i];
                Symbol state = null;
                _init_counter++;
                kwargs.Add("name", $"{_prefix}begin_state_{_init_counter}");
                if (info == null)
                {
                    info = new StateInfo(kwargs);
                }
                else
                {
                    info.Update(kwargs);
                }

                var obj = new sym();
                var m = typeof(sym).GetMethod(func.Replace("sym.", ""), BindingFlags.Static);
                var keys = m.GetParameters().Select(x => x.Name).ToArray();
                var paramArgs = info.GetArgs(keys);
                states.Add((Symbol)m.Invoke(obj, paramArgs));
            }

            return states;
        }

        public virtual NDArrayDict UnpackWeights(NDArrayDict args)
        {
            if (GateNames == null)
                return args;

            var h = _num_hidden;
            foreach (var group_name in new string[] { "i2h", "h2h" })
            {
                var weight = args[$"{_prefix}{group_name}_weight"];
                var bias = args[$"{_prefix}{group_name}_bias"];
                for (int j = 0; j < GateNames.Length; j++)
                {
                    var gate = GateNames[j];
                    string wname = $"{_prefix}{group_name}{gate}_weight";
                    args[wname] = weight[$"{j * h}:{(j + 1) * h}"].Copy();
                    string bname = $"{_prefix}{group_name}{gate}_bias";
                    args[bname] = weight[$"{j * h}:{(j + 1) * h}"].Copy();
                }
            }

            return args;
        }

        public virtual NDArrayDict PackWeights(NDArrayDict args)
        {
            if (this.GateNames == null)
            {
                return args;
            }
            foreach (var group_name in new List<string> {
                "i2h",
                "h2h"
            })
            {
                var weight = new List<NDArray>();
                var bias = new List<NDArray>();
                foreach (var gate in this.GateNames)
                {
                    var wname = $"{_prefix}{group_name}{gate}_weight";
                    weight.Add(args[wname]);
                    var bname = $"{_prefix}{group_name}{gate}_bias";
                    bias.Add(args[bname]);
                }

                args[$"{_prefix}{group_name}_weight"] = nd.Concat(weight);
                args[$"{_prefix}{group_name}_bias"] = nd.Concat(bias);
            }
            return args;
        }

        public virtual (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = "NTC", bool? merge_outputs = null)
        {
            this.Reset();
            var _tup_1 = __internals__.NormalizeSequence(length, inputs, layout, false);
            inputs = _tup_1.Item1;
            if (begin_state == null)
            {
                begin_state = this.BeginState();
            }
            var states = begin_state;
            var outputs = new SymbolList();
            foreach (var i in Enumerable.Range(0, length))
            {
                var _tup_2 = Call(inputs[i], states);
                var output = _tup_2.Item1;
                states = _tup_2.Item2;
                outputs.Add(output);
            }

            var _tup_3 = __internals__.NormalizeSequence(length, outputs, layout, merge_outputs.Value);
            outputs = _tup_3.Item1;
            return (outputs, states);
        }

        public Symbol GetActivation(Symbol inputs, ActivationType activation, string name)
        {
            return sym.Activation(inputs, activation, name);
        }
    }
}
