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
using MxNet.IO;
using System.Linq;

namespace MxNet.Modules
{
    public class ExecutorGroup
    {
        internal static void LoadGeneral(List<NDArrayList> data, List<NDArrayList> targets, int[] major_axis)
        {
            for (int i = 0; i < data.Count; i++)
            {
                var d_src = data[i];
                var d_targets = targets[i];
                int axis = major_axis[i];
                for (int j = 0; j < d_src.Length; j++)
                {
                    var src = d_src[j];
                    var dst = d_targets[j];
                    src.CopyTo(dst);
                }
            }
        }

        internal static void LoadData(DataBatch batch, List<List<(Slice, NDArray)>> targets, int[] major_axis)
        {
            var datalist = new List<NDArrayList>() { batch.Data };
            List<NDArrayList> targetlist = new List<NDArrayList>();
            foreach (var item in targets)
            {
                targetlist.Add(item.Select(x => x.Item2).ToArray());
            }

            LoadGeneral(datalist, targetlist, major_axis);
        }

        internal static void LoadLabel(DataBatch batch, List<List<(Slice, NDArray)>> targets, int[] major_axis)
        {
            var datalist = new List<NDArrayList>() { batch.Label };
            List<NDArrayList> targetlist = new List<NDArrayList>(); 
            foreach (var item in targets)
            {
                targetlist.Add(item.Select(x => x.Item2).ToArray());
            }

            LoadGeneral(datalist, targetlist, major_axis);
        }

        internal static NDArrayList MergeMultiContext(List<NDArrayList> outputs, int[] major_axis)
        {
            var ret = Enumerable.Zip(outputs, major_axis, (tensors, axis) =>
            {
                if(axis >= 0)
                {
                    if (tensors.Length == 1)
                        return tensors[0];
                    else
                    {
                        return nd.Concat(tensors.Select(x => x.AsInContext(tensors[0].Context)).ToArray(), dim: axis);
                    }
                }

                return tensors[0];

            }).ToList();

            return ret;
        }

        internal static Dictionary<string, Context>[] PrepareGroup2Ctxs(Dictionary<string, Context>[] group2ctxs, int ctx_len)
        {
            List<Dictionary<string, Context>> ret = new List<Dictionary<string, Context>>();
            if(group2ctxs == null)
            {
                for (int i = 0; i < ctx_len; i++)
                    ret.Add(null);

                return ret.ToArray();
            }

            if (group2ctxs.Length == ctx_len)
                return group2ctxs;

            throw new Exception("Length of group2ctxs should be " + ctx_len);
        }
    }
}