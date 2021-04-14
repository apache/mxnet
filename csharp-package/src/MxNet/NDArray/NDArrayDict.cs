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
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using NDArrayHandle = System.IntPtr;
using MxNet.Interop;
using MxNet.Numpy;

namespace MxNet
{
    public class NDArrayDict : IEnumerable<KeyValuePair<string, ndarray>>
    {
        private readonly Dictionary<string, ndarray> dict = new Dictionary<string, ndarray>();

        public NDArrayDict(params string[] names)
        {
            foreach (var item in names) Add(item, null);
        }

        public int Count => dict.Count;

        public string[] Keys => dict.Keys.ToArray();

        public NDArrayList Values => dict.Values.ToArray();

        public ndarray this[string name]
        {
            get
            {
                if (!dict.ContainsKey(name))
                    return null;

                return dict[name];
            }
            set => dict[name] = value;
        }

        public IEnumerator<KeyValuePair<string, ndarray>> GetEnumerator()
        {
            return dict.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return dict.GetEnumerator();
        }

        public void Add(string name, ndarray value)
        {
            dict.Add(name, value);
        }

        public void Add(NDArrayDict other)
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

        public void Update(NDArrayDict dict)
        {
            foreach (var item in dict)
            {
                this[item.Key] = item.Value;
            }
        }

        public static NDArrayDict LoadFromBuffer(byte[] buffer)
        {
            LoadFromBuffer(buffer, out var data);
            return data;
        }

        public static void LoadFromBuffer(byte[] buffer, out NDArrayDict data)
        {
            data = new NDArrayDict();
            uint outSize;
            IntPtr outArrPtr;
            uint outNameSize;
            IntPtr outNamesPtr;
            NativeMethods.MXNDArrayLoadFromBuffer(buffer, buffer.Length, out outSize, out outArrPtr, out outNameSize, out outNamesPtr);

            var outArr = new NDArrayHandle[outSize];
            Marshal.Copy(outArrPtr, outArr, 0, (int)outSize);


            if (outNameSize == 0)
            {
                for (var i = 0; i < outArr.Length; i++) data.Add(i.ToString(), new ndarray(outArr[i]));
            }
            else
            {
                var outNames = new IntPtr[outNameSize];
                Marshal.Copy(outNamesPtr, outNames, 0, (int)outNameSize);

                for (var i = 0; i < outArr.Length; i++)
                {
                    var key = Marshal.PtrToStringAnsi(outNames[i]);
                    if (!string.IsNullOrEmpty(key)) data.Add(key, new ndarray(outArr[i]));
                }
            }
        }
    }
}