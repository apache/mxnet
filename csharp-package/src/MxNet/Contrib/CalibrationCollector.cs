using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib
{
    public abstract class CalibrationCollector
    {
        public CalibrationCollector()
        {
            throw new NotImplementedRelease2Exception();
        }

        public abstract void Collect(string name, string op_name, NDArray arr);

        public virtual Dictionary<string, float> PostCollect()
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
