using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class NotImplementedRelease1Exception : Exception
    {
        public NotImplementedRelease1Exception() : base("This function will be implemented in v2.0 release 1")
        {
        }
    }
}
