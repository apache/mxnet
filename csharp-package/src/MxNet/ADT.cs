using MxNet._ffi;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class ADT : MXNetObject
    {
        public ADT(int tag, MxObject[] fields)
        {
            throw new NotImplementedException();
        }

        public int Tag
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public (MxObject[], int, int) this[int idx]
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public int Length
        {
            get
            {
                throw new NotImplementedException();
            }
        }
    }
}
