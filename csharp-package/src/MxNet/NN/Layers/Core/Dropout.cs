using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class Dropout : BaseLayer
    {
        public float Rate { get; set; }

        public DropoutMode Mode { get; set; }

        public Dropout(float rate, DropoutMode mode = DropoutMode.Training)
            :base("dropout")
        {

        }

        public override Symbol Build(Symbol data)
        {
            return sym.Dropout(data, Rate, Mode, symbol_name: ID);
        }
        
    }
}
