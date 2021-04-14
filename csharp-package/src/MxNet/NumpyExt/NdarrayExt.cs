using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MxNet
{
    public static class NdarrayExt
    {
        public static T asscalar<T>(this ndarray source)
        {
            return source.ToArray<T>().FirstOrDefault();
        }

    }
}
