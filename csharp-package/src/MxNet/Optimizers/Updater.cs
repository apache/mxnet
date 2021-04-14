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
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MxNet.Optimizers
{
    public class Updater
    {
        private readonly bool aggregate_updates;
        internal Optimizer optimizer;
        internal Dictionary<int, (NDArrayDict, NDArray)> states;
        private readonly Dictionary<int, bool> states_synced;

        public Updater(Optimizer opt)
        {
            optimizer = opt;
            states = new Dictionary<int, (NDArrayDict, NDArray)>();
            states_synced = new Dictionary<int, bool>();
            aggregate_updates = opt.AggregateNum > 0;
        }

        public void Call(int index, NDArray grad, NDArray weight)
        {
            Call(new[] {index}, grad, weight);
        }

        public void Call(int[] indices, NDArrayList grads, NDArrayList weights)
        {
            if (weights != null)
                optimizer.SetCurrentContext(weights[0].ctx.GetDeviceId());

            for (var i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                if (!states.ContainsKey(index))
                {
                    states[index] = optimizer.CreateStateMultiPrecision(index, weights[i]);
                    states_synced[index] = true;
                }
                else if (!states_synced[index])
                {
                    states[i] = SyncStateContext(states[i], weights[i].ctx);
                    states_synced[index] = true;
                }
            }

            if (aggregate_updates)
            {
                var type_map = new Dictionary<string, List<(int, NDArray, NDArray)>>();
                for (var i = 0; i < indices.Length; i++)
                {
                    var w = weights[i];
                    var g = grads[i];
                    if (type_map.ContainsKey(w.dtype.Name))
                    {
                        type_map[w.dtype.Name].Add((i, w, g));
                    }
                    else
                    {
                        type_map[w.dtype.Name] = new List<(int, NDArray, NDArray)>();
                        type_map[w.dtype.Name].Add((i, w, g));
                    }
                }

                foreach (var item in type_map)
                {
                    var idx = item.Key;
                    var current_index = 0;
                    (indices, weights, grads) = (item.Value.Select(x => (x.Item1)).ToArray(), item.Value.Select(x => (x.Item2)).ToArray(),                                  item.Value.Select(x => (x.Item3)).ToArray());

                    while (current_index < indices.Length)
                    {
                        var local_states = new Dictionary<int, (NDArrayDict, ndarray)>();
                        var step = Math.Min(optimizer.AggregateNum, indices.Length - current_index);
                        
                        for (var j = 0; j < step; j++) 
                            local_states.Add(j, states[indices[current_index + j]]);

                        var forupdate = item.Value.Skip(current_index).Take(current_index + optimizer.AggregateNum).ToArray();
                        var (index, weight, grad) = (forupdate.Select(x => (x.Item1)).ToArray(), forupdate.Select(x => (x.Item2)).ToArray(), forupdate.Select(x => (x.Item3)).ToArray());
                        optimizer.UpdateMultiPrecision(index, weight, grad, local_states.Values.ToArray()); //ToDo: revisit code

                        current_index += optimizer.AggregateNum;
                    }
                }
            }
            else
            {
                for (var i = 0; i < indices.Length; i++)
                    optimizer.UpdateMultiPrecision(indices[i], weights[i], grads[i], states[i]);
            }
        }

        public (NDArrayDict, NDArray) SyncStateContext((NDArrayDict, NDArray) state, Context context)
        {
            var (dict, arr) = state;
            var dict1 = new NDArrayDict();
            foreach (var item in dict) dict1[item.Key] = item.Value.AsInContext(context);

            return (dict1, arr.AsInContext(context));
        }

        public void SetStates(string states_data)
        {
            var (states, opt) = Pickle.Loads(states_data);
            if (opt != null)
                optimizer = opt;

            this.states = states;
            states_synced.Clear();
            foreach (var item in states.Keys) states_synced.Add(item, false);
        }

        public string GetStates(bool dump_optimizer = false)
        {
            if (dump_optimizer)
                return Pickle.Dumps(states, optimizer);

            return Pickle.Dumps(states);
        }
    }
}