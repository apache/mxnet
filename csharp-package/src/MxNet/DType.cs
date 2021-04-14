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
using System.Collections.Generic;

namespace MxNet
{
    public class DType
    {
        private static readonly Dictionary<string, DType> StringToDTypeMap = new Dictionary<string, DType>();
        private static readonly Dictionary<int, DType> IndexToDTypeMap = new Dictionary<int, DType>();
        public static readonly DType Float32 = new DType("float32", "Single", 0);
        public static readonly DType Float64 = new DType("float64", "Double", 1);
        public static readonly DType Float16 = new DType("float16", "Single", 2);
        public static readonly DType UInt8 = new DType("uint8", "Byte", 3);
        public static readonly DType Int32 = new DType("int32", "Int32", 4);
        public static readonly DType Int8 = new DType("int8", "SByte", 5);
        public static readonly DType Int64 = new DType("int64", "Int64", 6);
        public static readonly DType Bool = new DType("bool", "Boolean", 7);
        public static readonly DType Int16 = new DType("int16", "Int16", 8);
        public static readonly DType UInt16 = new DType("uint16", "UInt16", 9);
        public static readonly DType UInt32 = new DType("uint32", "UInt32", 10);
        public static readonly DType UInt64 = new DType("uint64", "UInt64", 11);

        public DType(string name, string csName, int index)
        {
            Name = name;
            CsName = csName;
            Index = index;
            StringToDTypeMap.Add(Name, this);
            IndexToDTypeMap.Add(index, this);
        }

        public string Name { get; }
        public string CsName { get; }
        public int Index { get; }

        public static implicit operator string(DType value)
        {
            return value != null ? value.Name : "";
        }

        public static implicit operator DType(string value)
        {
            return StringToDTypeMap[value];
        }

        public static explicit operator DType(int index)
        {
            return IndexToDTypeMap[index];
        }

        public override string ToString()
        {
            return Name;
        }

        public static DType GetType(int index)
        {
            return IndexToDTypeMap[index];
        }

        public static DType InferDtype(Array data)
        {
            DType dtype = DType.Float32;
            if (data.GetType().Name.Contains("SByte"))
            {
                dtype = DType.Int8;
            }
            else if (data.GetType().Name.Contains("Byte"))
            {
                dtype = DType.UInt8;
            }
            else if (data.GetType().Name.Contains("Single"))
            {
                dtype = DType.Float32;
            }
            else if (data.GetType().Name.Contains("Double"))
            {
                dtype = DType.Float64;
            }
            else if (data.GetType().Name.Contains("Int32"))
            {
                dtype = DType.Int32;
            }
            else if (data.GetType().Name.Contains("Boolean"))
            {
                dtype = DType.Bool;
            }
            else if (data.GetType().Name.Contains("Int16"))
            {
                dtype = DType.Int16;
            }
            else if (data.GetType().Name.Contains("UInt16"))
            {
                dtype = DType.UInt16;
            }
            else if (data.GetType().Name.Contains("UInt32"))
            {
                dtype = DType.UInt32;
            }
            else if (data.GetType().Name.Contains("UInt64"))
            {
                dtype = DType.UInt64;
            }

            return dtype;
        }
    }
}