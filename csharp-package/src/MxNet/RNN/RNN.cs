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
using MxNet.Callbacks;
using MxNet.RecurrentLayer;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.RNN
{
    public class RNN
    {
        public static void SaveRNNCheckPoint(BaseRNNCell[] cells, string prefix, int epoch, Symbol symbol, NDArrayDict arg_params, NDArrayDict aux_params )
        {
            foreach (var cell in cells)
            {
                arg_params = cell.UnpackWeights(arg_params);
            }

            MxModel.SaveCheckpoint(prefix, epoch, symbol, arg_params, aux_params);
        }

        public static (Symbol, NDArrayDict, NDArrayDict) LoadRNNCheckPoint(BaseRNNCell[] cells, string prefix, int epoch)
        {
            var (sym, arg, aux) = MxModel.LoadCheckpoint(prefix, epoch);
            foreach (var cell in cells)
            {
                arg = cell.PackWeights(arg);
            }

            return (sym, arg, aux);
        }
    }

    public class RNNCheckPointCallback : IEpochEndCallback
    {
        private int _period;
        private string _prefix;
        private RNNCell[] _cells;

        public RNNCheckPointCallback(RNNCell[] cells, string prefix, int period = 1)
        {
            _cells = cells;
            _prefix = prefix;
            _period = period;
        }

        public void Invoke(int epoch, Symbol symbol, NDArrayDict arg_params, NDArrayDict aux_params)
        {
            if ((epoch + 1) % _period == 0)
                RNN.SaveRNNCheckPoint(_cells, _prefix, epoch, symbol, arg_params, aux_params);
        }
    }
}
