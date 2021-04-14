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

namespace MxNet.IO
{
    internal class IOUtils
    {
        public static NDArrayDict InitData(NDArrayList data, bool allow_empty, string default_name)
        {
            var result = new NDArrayDict();
            if (data == null)
                return result;

            if (!allow_empty && data.Length == 0) throw new Exception("Data cannot be empty when allow_empty is false");

            if (data.Length == 1)
                result.Add(default_name, data[0]);
            else
                for (var i = 0; i < data.Length; i++)
                    result.Add($"_{i}_{default_name}", data[i]);

            return result;
        }

        public static bool HasInstance(NDArrayDict data, DType dtype)
        {
            foreach (var item in data)
                if (item.Value.dtype.Name == dtype.Name)
                    return true;

            return false;
        }

        public static NDArrayDict GetDataByIdx(NDArrayDict data)
        {
            var shuffle_data = new NDArrayDict();

            foreach (var item in data) shuffle_data.Add(item.Key, nd.Shuffle(item.Value));

            return shuffle_data;
        }
    }
}