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
using MxNet.Libs;
using System.Collections.Generic;
using System.Threading;
using static MxNet.Name;

namespace MxNet.Gluon
{
    public class _BlockScope : MxDisposable
    {
        private static readonly ThreadLocal<_BlockScope> _current = new ThreadLocal<_BlockScope>();
        internal ContextVar<Dictionary<string, int>> _naming_counter = new ContextVar<Dictionary<string, int>>("namecounter");
        internal ContextVar<string> _prefix = new ContextVar<string>("prefix", "");
        private readonly Block _block;
        private readonly Dictionary<string, int> _counter = new Dictionary<string, int>();

        public _BlockScope(Block block)
        {
            _block = block;
            _counter = new Dictionary<string, int>();
        }

        public ParameterDict Params { get; set; }

        public override MxDisposable With()
        {
            var name = _block.GetType().Name.ToLower();
            var counter = _naming_counter.Get();
            if (counter != null)
            {
                var count = counter.ContainsKey(name) ? counter[name] : 0;
                if (!counter.ContainsKey(name))
                    counter.Add(name, 0);

                counter[name] = count + 1;
                name = $"{name}{count}";
            }

            var counter_token = _naming_counter.Set(new Dictionary<string, int>());
            var prefix_token = _prefix.Set(_prefix.Get() + name + "_");
            
            using (var n =  new Prefix(_prefix.Get())) {
                var p = Profiler.Scope(name + ":");
            }

            _naming_counter.Reset(counter_token);

            return this;
        }

        public override void Exit()
        {
            if (string.IsNullOrWhiteSpace(_block.GetType().Name.ToLower()))
                return;
        }
    }
}