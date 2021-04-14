using MxNet.Gluon.Data;
using MxNet.IO;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib
{
    public class DataLoaderIter : DataIter
    {
        public DataLoaderIter(DataLoader loader, string data_name= "data", string label_name= "softmax_label", DType dtype= null)
        {
            throw new NotImplementedRelease2Exception();
        }

        public override bool End()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override NDArrayList GetData()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override int[] GetIndex()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override NDArrayList GetLabel()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override int GetPad()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override bool IterNext()
        {
            throw new NotImplementedRelease2Exception();
        }

        public override void Reset()
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
