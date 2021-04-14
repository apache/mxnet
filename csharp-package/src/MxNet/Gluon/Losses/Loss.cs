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
namespace MxNet.Gluon.Losses
{
    public class Loss : HybridBlock
    {
        public Loss(float? weight = null, int? batch_axis = null, string prefix = "", ParameterDict @params = null) :
            base()
        {
            Weight = weight;
            BatchAxis = batch_axis;
        }

        public float? Weight { get; }
        public int? BatchAxis { get; internal set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            return HybridForward(x, args[0], args.Length > 1 ? args[1] : null);
        }

        public virtual NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            return pred;
        }

        public NDArrayOrSymbol ApplyWeighting(NDArrayOrSymbol loss, float? weight = null,
            NDArrayOrSymbol sample_weight = null)
        {
            if (sample_weight != null)
            {
                if (loss.IsNDArray)
                    loss = nd.BroadcastMul(loss, sample_weight);
                else
                    loss = sym.BroadcastMul(loss, sample_weight);
            }

            if (weight.HasValue)
            {
                if (loss.IsNDArray)
                    loss = loss.NdX * weight.Value;
                else
                    loss = loss.SymX * weight.Value;
            }

            return loss;
        }
    }
}