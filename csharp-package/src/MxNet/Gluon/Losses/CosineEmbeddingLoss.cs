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
using MxNet.Numpy;
using MxNet.Sym.Numpy;

namespace MxNet.Gluon.Losses
{
    public class CosineEmbeddingLoss : Loss
    {
        public CosineEmbeddingLoss(float? weight = null, int? batch_axis = null, float margin = 0, string prefix = null,
            ParameterDict @params = null) : base(weight, batch_axis)
        {
            Margin = margin;
        }

        public float Margin { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol input1, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            input1 = F.reshape_like(input1, label);
            label = F.reshape(label, new Shape(-1, 1));
            var cos_sim = _cosine_similarity(input1, label);
            var y_1 = F.equal(label, 1);
            var y_minus_1 = F.equal(label, -1);
            var cos_sim_a = (1 - cos_sim) * y_1;
            var z_array = np.zeros(new Shape(1, 1));
            var cos_sim_b =
                F.maximum(z_array, y_minus_1 * (cos_sim - Margin)); //ToDo: Check missing axis parameter
            var loss = cos_sim_a + cos_sim_b;
            loss = ApplyWeighting(loss, Weight, sample_weight);
            return loss;
        }

        private NDArrayOrSymbol _cosine_similarity(NDArrayOrSymbol x, NDArrayOrSymbol y, int axis = -1)
        {
            var x_norm = F.norm(x, axis: new Shape(axis)).Reshape(-1, 1);
            var y_norm = F.norm(y, axis: new Shape(axis)).Reshape(-1, 1);
            var x_dot_y = F.sum(x * y, axis).Reshape(-1, 1);
            var eps_err = new ndarray(new[] {1e-12f});
            return x_dot_y / F.maximum(x_norm * y_norm, eps_err);
        }
    }
}