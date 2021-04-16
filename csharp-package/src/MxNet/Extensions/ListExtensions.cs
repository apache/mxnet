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
using System.Linq;

namespace MxNet
{
    public static class ListExtensions
    {
        public static SymbolList ToSymbols(this NDArrayOrSymbolList source)
        {
            return source.Select(x => x.SymX).ToArray();
        }

        public static NDArrayList ToNDArrays(this NDArrayOrSymbolList source)
        {
            return source.Select(x => x.NdX).ToArray();
        }

        public static NDArrayOrSymbol Sum(this NDArrayOrSymbolList source)
        {
            NDArrayOrSymbol result = null;
            if (source.Length > 0)
            {
                if (source[0].IsNDArray)
                    foreach (var item in source)
                    {
                        if (result == null)
                        {
                            result = item;
                            continue;
                        }

                        result = result.NdX + item.NdX;
                    }
                else if (source[0].IsSymbol)
                    foreach (var item in source)
                    {
                        if (result == null)
                        {
                            result = item;
                            continue;
                        }

                        result = result.SymX + item.SymX;
                    }
            }

            return result;
        }

        public static string ToValueString(this List<float> source)
        {
            string line = string.Join(",", source);
            return $"({line})";
        }

        public static string ToValueString(this float[] source)
        {
            string line = string.Join(",", source);
            return $"({line})";
        }

        public static T Pop<T>(this List<T> source, int index)
        {
            int n = source.Count();
            if (index < 0)
                index = n - index;

            T value = source.ElementAt(index);
            source.RemoveAt(index);
            return value;
        }

    }
}