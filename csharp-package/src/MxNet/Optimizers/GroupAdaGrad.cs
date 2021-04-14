using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace MxNet.Optimizers
{
    public class GroupAdaGrad : Optimizer
    {
        public float Epsilon { get; set; }

        public GroupAdaGrad(float learning_rate= 0.01f, float epsilon= 1e-6f, bool use_fused_step= true) 
            : base(learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Epsilon = epsilon;
        }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict("history");
            state["history"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            return state;
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            var is_sparse = grad.stype == StorageStype.RowSparse;

            if (is_sparse)
            {
                this.UpdateCount(index);
                var lr = this.GetLr(index);
                var wd = this.GetWd(index);
                Debug.Assert(wd == 0, "Weight decay is not supported for GroupAdaGrad");

                weight = nd.Contrib.GroupAdagradUpdate(weight, grad, state["history"], lr, this.RescaleGrad, this.ClipGradient.HasValue ? this.ClipGradient.Value : -1, this.Epsilon);
            }
            else
            {
                Step(index, weight, grad, state);
            }
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            this.UpdateCount(index);
            var lr = this.GetLr(index);
            var wd = this.GetWd(index);
            Debug.Assert(wd == 0, "Weight decay is not supported for GroupAdaGrad");
            // preprocess grad
            grad = grad * this.RescaleGrad;
            if (this.ClipGradient != null)
            {
                grad = nd.Clip(grad, -this.ClipGradient.Value, this.ClipGradient.Value);
            }
            // update history
            state["history"] += np.mean(np.square(grad), axis: 1, keepdims: true);
            // update weight
            var d = grad / (np.sqrt(state["history"]) + this.Epsilon);
            weight -= lr * d;
        }
    }
}
