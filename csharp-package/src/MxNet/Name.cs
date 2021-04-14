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
using System.Threading;

namespace MxNet
{
    public class Name
    {
        public class NameManager : MxDisposable
        {
            internal static ThreadLocal<NameManager> current = new ThreadLocal<NameManager>();

            private readonly Dictionary<string, int> counter = new Dictionary<string, int>();
            private NameManager old_manager;

            public static NameManager Current
            {
                get
                {
                    if (current.IsValueCreated)
                        return current.Value;

                    return new NameManager();
                }
                set => current.Value = value;
            }

            public override MxDisposable With()
            {
                old_manager = Current;
                current.Value = this;
                return this;
            }

            public override void Exit()
            {
                if (old_manager != null) current.Value = old_manager;
            }

            public virtual string Get(string name, string hint)
            {
                if (!string.IsNullOrWhiteSpace(name)) return name;

                if (!counter.ContainsKey(hint)) counter[hint] = 0;

                name = hint + counter[hint];
                counter[hint]++;
                return name;
            }
        }

        public class Prefix : NameManager
        {
            private readonly string _prefix;

            public Prefix(string prefix)
            {
                _prefix = prefix;
            }

            public override string Get(string name, string hint)
            {
                var r = base.Get(name, hint);
                return _prefix + r;
            }
        }
    }
}