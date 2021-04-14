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
namespace MxNet.Gluon.NN
{
    public class ReflectionPad2D : HybridBlock
    {
        public ReflectionPad2D(int padding = 0) : base()
        {
            Padding = padding;
        }

        public int Padding { get; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            if (x.IsNDArray)
                return nd.Pad(x.NdX, PadMode.Reflect, new Shape(Padding));

            return sym.Pad(x.SymX, PadMode.Reflect, new Shape(Padding));
        }
    }
}