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

namespace MxNet.Initializers
{
    public class Load
    {
        public Load(NDArrayDict param, Initializer default_init = null, bool verbose = false)
        {
            Param = new NDArrayDict();
            foreach (var p in param)
                if (p.Key.StartsWith("arg:") || p.Key.StartsWith("aux:"))
                    Param[p.Key.Substring(4)] = p.Value;
                else
                    Param[p.Key] = p.Value;

            DefaultInit = default_init;
            Verbose = verbose;
        }

        public NDArrayDict Param { get; set; }

        public Initializer DefaultInit { get; set; }

        public bool Verbose { get; set; }

        public void Call(string name, ndarray arr)
        {
            if (Param.Contains(name))
            {
                if (arr.shape != Param[name].shape)
                    throw new MXNetException(string.Format("Shape mismatch, target {0} vs loaded {1}", arr.shape,
                        Param[name].shape));

                arr = Param[name];
                if (Verbose)
                    Logger.Log(string.Format("Initialized {0} by loading", name));
            }
            else
            {
                if (DefaultInit == null)
                    throw new MXNetException(string.Format(
                        "Cannot Initialize {0}. Not found in loaded param and no default Initializer is provided",
                        name));

                DefaultInit.InitWeight(name, ref arr);
                if (Verbose)
                    Logger.Log(string.Format("Initialized {0} by default", name));
            }
        }
    }
}