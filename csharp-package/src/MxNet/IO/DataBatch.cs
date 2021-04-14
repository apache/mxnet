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
namespace MxNet.IO
{
    /// <summary>
    ///     Default object for holding a mini-batch of data and related information. This class cannot be inherited.
    /// </summary>
    public sealed class DataBatch
    {
        public DataBatch(NDArrayList data, NDArrayList label = null, int? pad = null, int[] index = null,
            int? bucket_key = null, DataDesc[] provide_data = null, DataDesc[] provide_label = null)
        {
            Data = data;
            Label = label;
            Pad = pad;
            Index = index;
            BucketKey = bucket_key;
            ProvideData = provide_data;
            ProvideLabel = provide_label;
        }

        public DataBatch Shallowcopy()
        {
            return (DataBatch) MemberwiseClone();
        }

        #region Properties

        public NDArrayList Data { get; internal set; }

        public int[] Index { get; internal set; }

        public NDArrayList Label { get; internal set; }

        public int? Pad { get; internal set; }

        public DataDesc[] ProvideData { get; internal set; }

        public DataDesc[] ProvideLabel { get; internal set; }

        public int? BucketKey { get; internal set; }

        #endregion
    }
}