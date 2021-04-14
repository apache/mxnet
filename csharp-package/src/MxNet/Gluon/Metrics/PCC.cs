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
using System.Linq;
using System.Threading.Tasks;

namespace MxNet.Gluon.Metrics
{
    public class PCC : EvalMetric
    {
        private ndarray gcm;

        private int k;
        private ndarray lcm;

        public PCC(string output_name = null, string label_name = null)
            : base("pcc", output_name, label_name, true)
        {
            k = 2;
        }

        public float SumMetric => CalcMcc(lcm) * num_inst;

        public float GlobalSumMetric => CalcMcc(gcm) * global_num_inst;

        private void Grow(int inc)
        {
            lcm = nd.Pad(lcm, PadMode.Constant, new Shape(0, inc, 0, inc));
            gcm = nd.Pad(gcm, PadMode.Constant, new Shape(0, inc, 0, inc));
            k += inc;
        }

        private float CalcMcc(ndarray cmatArr)
        {
            var n = cmatArr.sum();
            var x = cmatArr.sum(1);
            var y = cmatArr.sum(0);
            var cov_xx = nd.Sum(x * (n - x)).AsScalar<float>();
            var cov_yy = nd.Sum(y * (n - y)).AsScalar<float>();

            if (cov_xx == 0 || cov_yy == 0)
                return float.NaN;

            var i = cmatArr.diag();
            var cov_xy = np.sum(i * n - x * y).sum().AsScalar<float>();
            return (float)System.Math.Pow(cov_xy / (cov_xx * cov_yy), 0.5);
        }

        public override void Update(ndarray labels, ndarray preds)
        {
            var pred = nd.Argmax(preds, 1).AsType(DType.Int32);
            var n = nd.Maximum(pred.Max(), labels.max()).AsScalar<int>();
            if (n >= k)
                Grow(n + 1 - k);

            var bcm = np.zeros(new Shape(k, k));
            var pred_data = pred.GetValues<int>();
            var label_data = labels.GetValues<int>();
            Enumerable.Zip(pred_data, label_data, (i, j) => {
                bcm[$"{i},{j}"] += 1;
                return true;
            });

            lcm += bcm;
            gcm += bcm;

            num_inst += 1;
            global_num_inst += 1;
        }

        public override void Reset()
        {
            global_num_inst = 0;
            gcm = nd.Zeros(new Shape(k, k));
            ResetLocal();
        }

        public override void ResetLocal()
        {
            num_inst = 0;
            lcm = nd.Zeros(new Shape(k, k));
        }
    }
}