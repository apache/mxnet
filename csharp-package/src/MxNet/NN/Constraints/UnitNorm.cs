using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Constraints
{
    public class UnitNorm : BaseConstraint
    {
        public uint Axis;

        public UnitNorm(uint axis = 0)
        {
            Axis = axis;
        }

        public override NDArray Call(NDArray w)
        {
            w = w / nd.Sqrt(nd.Sum(w, new Shape(Axis), true));
            return w;
        }
    }
}
