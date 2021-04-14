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
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MxNet
{
    public class NDArrayList : IEnumerable<ndarray>
    {
        public List<ndarray> data;

        public NDArrayList()
        {
            data = new List<ndarray>();
        }

        public NDArrayList(int length)
        {
            data = new List<ndarray>();
            for (int i = 0; i < length; i++)
                data.Add(new ndarray());
        }

        public NDArrayList(params ndarray[] args)
        {
            data = args.ToList();
        }

        public NDArrayList((ndarray, ndarray) args)
        {
            data = new List<ndarray> { args.Item1, args.Item2 };
        }

        public NDArrayList((ndarray, ndarray, ndarray) args)
        {
            data = new List<ndarray> { args.Item1, args.Item2, args.Item3 };
        }

        public void Deconstruct(out ndarray x0, out ndarray x1)
        {
            x0 = this[0];
            x1 = this[1];
        }

        public void Deconstruct(out ndarray x0, out ndarray x1, out ndarray x2)
        {
            x0 = this[0];
            x1 = this[1];
            x2 = this[2];
        }

        public void Deconstruct(out ndarray x0, out ndarray x1, out ndarray x2, out ndarray x3)
        {
            x0 = this[0];
            x1 = this[1];
            x2 = this[2];
            x3 = this[3];
        }

        public ndarray[] Data => data.ToArray();

        public IntPtr[] Handles
        {
            get
            {
                List<IntPtr> ret = new List<IntPtr>();
                foreach (var item in Data)
                {
                    if (item == null)
                        continue;

                    ret.Add(item.NativePtr);
                }

                return ret.ToArray();
            }

        }

        public NDArrayOrSymbol[] NDArrayOrSymbols => data.Select(x => new NDArrayOrSymbol(x)).ToArray();

        public ndarray this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }

        public int Length => data.Count;

        public IEnumerator<ndarray> GetEnumerator()
        {
            return data.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return data.GetEnumerator();
        }

        public void Add(params ndarray[] x)
        {
            if (x == null)
                return;

            data.AddRange(x);
        }

        public static implicit operator NDArrayList(ndarray[] x)
        {
            return new NDArrayList(x);
        }

        public static implicit operator NDArrayList(NDArray[] x)
        {
            return new NDArrayList(x.Select(x => new ndarray(x.NativePtr)).ToArray());
        }

        public static implicit operator NDArrayList(ndarray x)
        {
            return new NDArrayList(x);
        }

        public static implicit operator NDArrayList(NDArray x)
        {
            return new NDArrayList(x);
        }

        public static implicit operator NDArrayList(List<ndarray> x)
        {
            return new NDArrayList(x.ToArray());
        }

        public static implicit operator NDArrayList(NDArrayOrSymbol[] x)
        {
            return new NDArrayList(x.Select(i => i.NdX).ToArray());
        }

        public static implicit operator NDArrayList(NDArrayOrSymbol x)
        {
            return new NDArrayList(x);
        }

        public static implicit operator NDArrayList(List<NDArrayOrSymbol> x)
        {
            return new NDArrayList(x.Select(i => i.NdX).ToArray());
        }

        public static implicit operator ndarray(NDArrayList x)
        {
            return x.data.Count > 0 ? x[0] : null;
        }

        public static implicit operator List<ndarray>(NDArrayList x)
        {
            return x.data.ToList();
        }

        public static implicit operator ndarray[](NDArrayList x)
        {
            if (x == null)
                return null;

            return x.data.ToArray();
        }
    }
}