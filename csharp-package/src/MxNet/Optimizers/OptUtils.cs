using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Optimizers
{
    public class OptUtils
    {
        public static NDArrayList FlattenList(List<NDArrayList> nested_list)
        {
            NDArrayList result = new NDArrayList();
            foreach (var item in nested_list)
            {
                result.Add(item);
            }

            return result;
        }
    }
}
