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
    public class LSTM : RNNLayer
    {
        public LSTM(int hidden_size, int num_layers= 1, string layout= "TNC",
                 float dropout= 0, bool bidirectional= false, int input_size= 0,
                 Initializer i2h_weight_initializer= null, Initializer h2h_weight_initializer= null,
                 Initializer i2h_bias_initializer= null, Initializer h2h_bias_initializer= null,
                 int? projection_size= null, Initializer h2r_weight_initializer= null,
                 float? state_clip_min= null, float?  state_clip_max = null, bool state_clip_nan = false,
                 DType dtype= null) : 
                base(hidden_size, num_layers, layout, dropout, bidirectional, input_size, i2h_weight_initializer, 
                    h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, "lstm", projection_size, 
                    h2r_weight_initializer, state_clip_min, state_clip_max, state_clip_nan, dtype, false)
        {
            throw new NotImplementedException();
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            throw new NotImplementedException();
        }
    }
}
