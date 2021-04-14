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
using MxNet.Initializers;
using MxNet.Numpy;

namespace MxNet.Gluon
{
    public class Constant : Parameter
    {
        public Constant(string name, ndarray value) : base(name, OpGradReq.Null, value.shape, value.dtype,
            init: new CInit(value))
        {
            Value = value;
            InitName = $"Constant_{Name}_{GetHashCode()}";
        }

        public override OpGradReq GradReg
        {
            get => OpGradReq.Null;
            set
            {
                if (value != OpGradReq.Null)
                    throw new Exception("Constant parameter only support grad_req->null");
            }
        }

        public ndarray Value { get; set; }

        public string InitName { get; set; }

        public class CInit : Initializer
        {
            private readonly ndarray _value;

            public CInit(ndarray value)
            {
                _value = value;
            }

            public override void InitWeight(string name, ref ndarray arr)
            {
                _value.CopyTo(arr);
            }
        }
    }
}