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
using System;
using MxNet.Numpy;
using Newtonsoft.Json;

namespace MxNet.Initializers
{
    public abstract class Initializer
    {
        private Func<ndarray, string> print_func;
        private bool verbose;

        public Initializer()
        {
            verbose = false;
            print_func = null;
        }

        public void SetVerbosity(bool verbose = false, Func<ndarray, string> print_func = null)
        {
            this.verbose = verbose;
            if (print_func == null)
                print_func = x => { return (nd.Norm(x) / (float) Math.Sqrt(x.size)).AsScalar<float>().ToString(); };

            this.print_func = print_func;
        }

        public abstract void InitWeight(string name, ref ndarray arr);

        private void VerbosePrint(InitDesc desc, string init, ndarray arr)
        {
            if (verbose && print_func != null)
                Logger.Info(string.Format("Initialized {0} as {1}: {2}", desc, init, print_func(arr)));
        }

        public string Dumps()
        {
            return JsonConvert.SerializeObject(this);
        }

        public static implicit operator Initializer(string name)
        {
            return Get(name);
        }

        public static Initializer Get(string name)
        {
            if (name == null)
                name = "";
            switch (name.ToLower().Trim())
            {
                case "xavier":
                    return new Xavier();
                case "zeros":
                    return new Zero();
                case "ones":
                    return new One();
                default:
                    return null;
            }
        }
    }
}