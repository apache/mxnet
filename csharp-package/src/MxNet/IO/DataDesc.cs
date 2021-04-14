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

namespace MxNet.IO
{
    public class DataDesc
    {
        public DataDesc(string name, Shape shape, DType dtype = null, string layout = "NCHW'")
        {
            Name = name;
            Shape = shape;
            DataType = dtype ?? DType.Float32;
            Layout = layout;
        }

        public string Name { get; set; }

        public Shape Shape { get; set; }

        public DType DataType { get; set; }

        public string Layout { get; set; }

        public override string ToString()
        {
            return string.Format("DataDesc[{0}, {1}, {2}, {3}]", Name, Shape, DataType, Layout);
        }

        public static int GetBatchAxis(string layout)
        {
            if (string.IsNullOrWhiteSpace(layout))
                return 0;

            return layout.ToCharArray().ToList().FindIndex(x => x == 'N');
        }

        public static DataDesc[] GetList(Dictionary<string, Shape> shapes, Dictionary<string, DType> types = null)
        {
            var result = new List<DataDesc>();

            if (types != null)
                foreach (var item in shapes)
                    result.Add(new DataDesc(item.Key, item.Value, types[item.Key]));
            else
                foreach (var item in shapes)
                    result.Add(new DataDesc(item.Key, item.Value));

            return result.ToArray();
        }
    }
}