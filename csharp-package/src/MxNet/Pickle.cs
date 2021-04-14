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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MxNet.Optimizers;
using Newtonsoft.Json;

namespace MxNet
{
    public class PickedStates
    {
        public PickedStates()
        {
            States = new List<KeyValuePair<int, string>>();
        }

        public List<KeyValuePair<int, string>> States { get; set; }

        public Optimizer Optimizer { get; set; }
    }

    public class Pickle
    {
        public static string Dumps(Dictionary<int, (NDArrayDict, NDArray)> states, Optimizer optimizer = null)
        {
            var result = "";
            var pickedStates = new PickedStates();
            var states_dump = states.ToList();
            var dict = new Dictionary<int, NDArrayDict>();
            foreach (var item in states)
            {
                var v = item.Value.Item1;
                v["_state_item2_"] = item.Value.Item2;
                dict.Add(item.Key, v);
            }

            foreach (var item in dict)
            {
                NDArray.Save("temp.txt", item.Value);
                pickedStates.States.Add(new KeyValuePair<int, string>(item.Key, File.ReadAllText("temp.txt")));
                File.Delete("temp.txt");
            }

            pickedStates.Optimizer = optimizer;
            result = JsonConvert.SerializeObject(pickedStates);
            return result;
        }

        public static (Dictionary<int, (NDArrayDict, NDArray)>, Optimizer) Loads(string data,
            bool load_optimizer = false)
        {
            var pickedStates = JsonConvert.DeserializeObject<PickedStates>(data);
            var states = new Dictionary<int, (NDArrayDict, NDArray)>();

            foreach (var item in pickedStates.States)
            {
                NDArrayDict item1 = null;
                File.AppendAllText("temp.txt", item.Value);
                item1 = NDArray.Load("temp.txt");
                var item2 = item1["_state_item2_"];
                item1.Remove("_state_item2_");

                states.Add(item.Key, (item1, item2));
            }

            if (load_optimizer)
                return (states, pickedStates.Optimizer);

            return (states, null);
        }
    }
}