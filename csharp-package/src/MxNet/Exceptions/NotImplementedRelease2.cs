using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class NotImplementedRelease2Exception : Exception
    {
        public NotImplementedRelease2Exception() : base("This function will be implemented in v2.0 release 2")
        {
        }
    }
}
