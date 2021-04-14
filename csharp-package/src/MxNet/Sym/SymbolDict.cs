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
using MxNet.Sym.Numpy;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MxNet
{
    public class SymbolDict : IEnumerable<KeyValuePair<string, _Symbol>>
    {
        private readonly Dictionary<string, _Symbol> dict = new Dictionary<string, _Symbol>();

        public SymbolDict(params string[] names)
        {
            foreach (var item in names) Add(item, null);
        }

        public int Count => dict.Count;

        public string[] Keys => dict.Keys.ToArray();

        public SymbolList Values => dict.Values.ToArray();

        public _Symbol this[string name]
        {
            get
            {
                if (!dict.ContainsKey(name))
                    return null;

                return dict[name];
            }
            set => dict[name] = value;
        }

        public IEnumerator<KeyValuePair<string, _Symbol>> GetEnumerator()
        {
            return dict.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return dict.GetEnumerator();
        }

        public void Add(string name, _Symbol value)
        {
            dict.Add(name, value);
        }

        public void Add(SymbolDict other)
        {
            foreach (var item in other) Add(item.Key, item.Value);
        }

        public bool Contains(string name)
        {
            return dict.ContainsKey(name);
        }

        public void Remove(string name)
        {
            dict.Remove(name);
        }
    }
}