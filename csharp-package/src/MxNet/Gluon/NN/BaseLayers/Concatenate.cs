using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.NN
{
    public class Concatenate : Sequential
    {
        public int axis { get; set; }
        public Concatenate(int axis = -1)
        {
            this.axis = axis;
        }

        public override NDArrayOrSymbolList Forward(NDArrayOrSymbolList args)
        {
            var @out = new NDArrayOrSymbolList();
            foreach (var block in this._childrens.Values)
            {
                @out.Add(block.Call(args));
            }

            return F.concatenate(@out, axis: this.axis);
        }
    }
}
