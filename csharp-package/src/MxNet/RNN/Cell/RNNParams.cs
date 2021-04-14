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
using MxNet.Initializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class RNNParams
    {
        internal string _prefix;
        internal SymbolDict _params;

        public RNNParams(string prefix = "")
        {
            _prefix = prefix;
            _params = new SymbolDict();
        }

        public Symbol Get(string name, Dictionary<string, string> attr = null, Shape shape = null,
                            float? lr_mult = null, float? wd_mult = null, DType dtype = null, 
                            Initializer init = null, StorageStype? stype = null)
        {
            name = _prefix + name;
            if(!_params.Contains(name))
            {
                _params[name] = Symbol.Var(name, attr, shape, lr_mult, wd_mult, dtype, init, stype);
            }

            return _params[name];
        }
    }
}
