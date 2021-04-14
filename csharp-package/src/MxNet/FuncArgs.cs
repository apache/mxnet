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
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MxNet
{
    public class FuncArgs : IEnumerable<KeyValuePair<string, object>>
    {
        private readonly Dictionary<string, object> args = new Dictionary<string, object>();

        public object this[string name]
        {
            get
            {
                if (!args.ContainsKey(name))
                    return null;

                return args[name];
            }
            set => args[name] = value;
        }

        public object[] Values => args.Values.ToArray();

        public IEnumerator<KeyValuePair<string, object>> GetEnumerator()
        {
            return args.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public bool Contains(string key)
        {
            return args.ContainsKey(key);
        }

        public void Add(string name, object value)
        {
            args.Add(name, value);
        }

        public void Remove(string name)
        {
            args.Remove(name);
        }

        public T Get<T>(string name)
        {
            if (!args.ContainsKey(name))
                return default;

            return (T) args[name];
        }

        public T Get<T>(string name, T val)
        {
            if (!args.ContainsKey(name))
                return val;

            return (T)args[name];
        }
    }
}