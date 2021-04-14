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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MxNet.Interop;
using DataIterHandle = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public sealed class MXDataIterMap
    {
        #region Fields

        private readonly Dictionary<string, DataIterHandle> _DataIterCreators;

        #endregion

        #region Constructors

        public MXDataIterMap()
        {
            var r = NativeMethods.MXListDataIters(out var numDataIterCreators, out var dataIterCreators);
            Logging.CHECK_EQ(r, 0);


            _DataIterCreators = new Dictionary<string, DataIterHandle>((int) numDataIterCreators);

            var array = InteropHelper.ToPointerArray(dataIterCreators, numDataIterCreators);
            for (var i = 0; i < numDataIterCreators; i++)
            {
                r = NativeMethods.MXDataIterGetIterInfo(array[i],
                    out var name,
                    out var description,
                    out var num_args,
                    out var arg_names2,
                    out var arg_type_infos2,
                    out var arg_descriptions2);

                Logging.CHECK_EQ(r, 0);

                var str = Marshal.PtrToStringAnsi(name);
                _DataIterCreators.Add(str, array[i]);
            }
        }

        #endregion

        #region Methods

        public DataIterHandle GetMXDataIterCreator(string name)
        {
            return _DataIterCreators[name];
        }

        #endregion
    }
}