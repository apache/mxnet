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
using mx_float = System.Single;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public static class sym_ops
    {
        public static Symbol Plus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Plus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Sum(this Symbol lhs)
        {
            return new Operator("_Plus").Set(lhs).CreateSymbol();
        }

        public static Symbol Mul(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mul").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Minus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Div(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Div").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Mod(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mod").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Power(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Power").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Maximum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Maximum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Minimum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minimum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol Log(Symbol data)
        {
            return new Operator("log").SetInput("data", data).CreateSymbol();
        }

        public static Symbol PlusScalar(Symbol lhs, float scalar)
        {
            return new Operator("_PlusScalar").Set(lhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol MinusScalar(Symbol lhs, float scalar)
        {
            return new Operator("_MinusScalar").Set(lhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol RMinusScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RMinusScalar").Set(rhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol MulScalar(Symbol lhs, float scalar)
        {
            return new Operator("_MulScalar").Set(lhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol DivScalar(Symbol lhs, float scalar)
        {
            return new Operator("_DivScalar").Set(lhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol RDivScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RDivScalar").Set(rhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol ModScalar(Symbol lhs, float scalar)
        {
            return new Operator("_ModScalar").Set(lhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }

        public static Symbol RModScalar(float scalar, Symbol rhs)
        {
            return new Operator("_RModScalar").Set(rhs)
                .SetParam("scalar", scalar)
                .CreateSymbol();
        }
    }
}