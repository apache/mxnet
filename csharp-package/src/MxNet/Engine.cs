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
using MxNet.Interop;

namespace MxNet
{
    public class Engine
    {
        public static int SetBulkSize(int size)
        {
            var prev = 0;
            NativeMethods.MXEngineSetBulkSize(size, ref prev);
            return prev;
        }

        public static _BulkScope Bulk(int size)
        {
            return new _BulkScope(size);
        }

        public class _BulkScope : MxDisposable
        {
            private int _old_size;
            private readonly int _size;

            public _BulkScope(int size)
            {
                _size = size;
            }

            public override MxDisposable With()
            {
                _old_size = SetBulkSize(_size);
                return this;
            }

            public override void Exit()
            {
                SetBulkSize(_old_size);
            }
        }
    }
}