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
using System.Text.RegularExpressions;

namespace MxNet.Initializers
{
    public class Mixed
    {
        private readonly List<KeyValuePair<Regex, Initializer>> map = new List<KeyValuePair<Regex, Initializer>>();

        public Mixed(string[] patterns, Initializer[] initializers)
        {
            if (patterns.Length != initializers.Length)
                throw new ArgumentException("patterns and initializers not of same length");

            for (var i = 0; i < patterns.Length; i++)
                map.Add(new KeyValuePair<Regex, Initializer>(new Regex(patterns[i]), initializers[i]));
        }

        public void Call(string name, ndarray arr)
        {
            foreach (var item in map)
                if (item.Key.IsMatch(name))
                    item.Value.InitWeight(name, ref arr);
        }
    }
}