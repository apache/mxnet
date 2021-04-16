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
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MxNet
{
    public class SymbolList : IEnumerable<_Symbol>
    {
        public List<_Symbol> data;

        public SymbolList()
        {
            data = new List<_Symbol>();
        }

        public SymbolList(int length)
        {
            data = new List<_Symbol>();
            for (int i = 0; i < length; i++)
                data.Add(new _Symbol());
        }

        public SymbolList(params _Symbol[] args)
        {
            data = args.ToList();
        }

        public SymbolList((_Symbol, _Symbol) args)
        {
            data = new List<_Symbol> { args.Item1, args.Item2 };
        }

        public SymbolList((_Symbol, _Symbol, _Symbol) args)
        {
            data = new List<_Symbol> { args.Item1, args.Item2, args.Item3 };
        }

        public _Symbol[] Data => data.ToArray();

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

        public NDArrayOrSymbolList NDArrayOrSymbol => data.Select(x => new NDArrayOrSymbol(x)).ToList();

        public _Symbol this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }

        public int Length => data.Count;

        public IEnumerator<_Symbol> GetEnumerator()
        {
            return data.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return data.GetEnumerator();
        }

        public void Add(params _Symbol[] x)
        {
            if (x == null)
                return;

            data.AddRange(x);
        }

        public static implicit operator SymbolList(_Symbol[] x)
        {
            return new SymbolList(x);
        }

        public static implicit operator SymbolList(_Symbol x)
        {
            return new SymbolList(x);
        }

        public static implicit operator SymbolList(List<_Symbol> x)
        {
            return new SymbolList(x.ToArray());
        }

        public static implicit operator SymbolList(NDArrayOrSymbolList x)
        {
            return new SymbolList(x.Select(i => i.SymX).ToArray());
        }

        public static implicit operator SymbolList(NDArrayOrSymbol x)
        {
            return new SymbolList(x.SymX);
        }

        public static implicit operator _Symbol(SymbolList x)
        {
            return x.data.Count > 0 ? x[0] : null;
        }

        public static implicit operator List<_Symbol>(SymbolList x)
        {
            return x.data.ToList();
        }

        public static implicit operator _Symbol[](SymbolList x)
        {
            if (x == null)
                return null;

            return x.data.ToArray();
        }
    }
}