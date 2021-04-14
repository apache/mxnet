using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers.Recurrent
{
    public class RNN : BaseLayer
    {
        public RNN() : base("rnn")
        {

        }

        public override Symbol Build(Symbol x)
        {
            throw new NotImplementedException();
        }
    }
}
