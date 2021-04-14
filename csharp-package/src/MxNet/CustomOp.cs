using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public abstract class CustomOp
    {
        public abstract void Forward(bool is_train, OpGradReq[] req, NDArrayList in_data, NDArrayList out_data, NDArrayList aux);

        public abstract void Backward(OpGradReq[] req, NDArrayList out_grad, NDArrayList in_data, NDArrayList out_data, NDArrayList in_grad, NDArrayList aux);

        public virtual void Assign(ndarray dst, OpGradReq req, ndarray src)
        {
            if (req == OpGradReq.Null)
                return;
            else if (req == OpGradReq.Write)
                dst = src;
            else if (req == OpGradReq.Add)
                dst += src;
        }
    }
}
