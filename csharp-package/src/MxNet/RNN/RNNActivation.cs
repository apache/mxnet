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
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class RNNActivation
    {
        private string _name;
        private FuncArgs _kwargs;
        public RNNActivation(string name = "")
        {
            _name = name;
        }

        public Symbol Invoke(Symbol x, string name = "")
        {
            _name = name != "" ? name : _name;
            Symbol ret = null;

            switch (_name)
            {
                case "leaky":
                    ret = sym.LeakyReLU(x, symbol_name: _name);
                    break;
                case "relu":
                    ret = sym.Relu(x, symbol_name: _name);
                    break;
                default:
                    break;
            }

            return ret;
        }
    }
}
