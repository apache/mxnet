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
using static MxNet.Gluon.Block;

namespace MxNet.Gluon
{
    public class HookHandle : MxDisposable
    {
        private WeakReference<Dictionary<int, Hook>> _hooks_dict_ref;
        private int _id;

        public HookHandle()
        {
            _hooks_dict_ref = null;
        }

        public (WeakReference<Dictionary<int, Hook>>, int) State
        {
            get => (_hooks_dict_ref, _id);
            set
            {
                if (value.Item1 == null)
                    _hooks_dict_ref = new WeakReference<Dictionary<int, Hook>>(new Dictionary<int, Hook>());
                else
                    _hooks_dict_ref = value.Item1;

                _id = value.Item2;
            }
        }

        public void Attach(Dictionary<int, Hook> hooks_dict, Hook hook)
        {
            if (_hooks_dict_ref != null)
                throw new Exception("The same handle cannot be attached twice.");

            _id = hook.GetHashCode();
            hooks_dict[_id] = hook;
            _hooks_dict_ref = new WeakReference<Dictionary<int, Hook>>(hooks_dict);
        }

        public void Detach()
        {
            _hooks_dict_ref.TryGetTarget(out var hooks_dict);
            if (hooks_dict != null && hooks_dict.ContainsKey(_id))
                hooks_dict.Remove(_id);
        }

        public override MxDisposable With()
        {
            return this;
        }

        public override void Exit()
        {
            Detach();
        }
    }
}