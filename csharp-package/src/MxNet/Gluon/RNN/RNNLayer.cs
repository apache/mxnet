/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using MxNet.Gluon.RNN;
using MxNet.Initializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RecurrentNN
{
    public abstract class RNNLayer : HybridBlock
    {
        public RNNLayer(int hidden_size, int num_layers, string layout, float dropout, bool bidirectional, int input_size,
                 Initializer i2h_weight_initializer, Initializer h2h_weight_initializer, Initializer i2h_bias_initializer, 
                 Initializer h2h_bias_initializer, string mode, int? projection_size, Initializer h2r_weight_initializer,
                 float? lstm_state_clip_min, float? lstm_state_clip_max, bool? lstm_state_clip_nan,
                 DType dtype, bool use_sequence_length= false) : base()
        {
            throw new NotImplementedException();
        }

        private void RegisterParam(string name, Shape shape, Initializer init, DType dtype)
        {
            throw new NotImplementedException();
        }

        public abstract StateInfo[] StateInfo(int batch_size = 0);

        public override void Cast(DType dtype)
        {
            throw new NotImplementedException();
        }

        public virtual NDArrayOrSymbol[] BeginState(int batch_size = 0, string func = null, FuncArgs args = null)
        {
            throw new NotImplementedException();
        }

        public NDArrayOrSymbol Call(NDArrayOrSymbol inputs, NDArrayOrSymbol[] states, NDArrayOrSymbol sequence_length)
        {
            throw new NotImplementedException();
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            throw new NotImplementedException();
        }

        private (NDArrayOrSymbol, NDArrayOrSymbol[]) ForwardKernel(NDArrayOrSymbol inputs, NDArrayOrSymbol[] states, NDArrayOrSymbol sequence_length)
        {
            throw new NotImplementedException();
        }
    }
}
