using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Transformations
{
    public abstract class Transformation
    {
        public virtual NDArrayOrSymbol F { get; set; }

        public virtual NDArrayOrSymbol Sign
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public virtual NDArrayOrSymbol Inv
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public Transformation()
        {
            throw new NotImplementedRelease1Exception();
        }

        public virtual NDArrayOrSymbol Call(NDArrayOrSymbol x)
        {
            return ForwardCompute(x);
        }

        public virtual NDArrayOrSymbol InvCall(NDArrayOrSymbol x)
        {
            return InverseCompute(x);
        }

        
        public abstract NDArrayOrSymbol ForwardCompute(NDArrayOrSymbol x);

        public abstract NDArrayOrSymbol InverseCompute(NDArrayOrSymbol x);

        public abstract NDArrayOrSymbol LogDetJacobian(NDArrayOrSymbol x, NDArrayOrSymbol y);
    }
}
