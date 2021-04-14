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
using System;
using System.Collections.Generic;

namespace MxNet.Optimizers
{
    public class CumGrad
    {
        public CumGrad(int numCums, NDArray cumGrad)
        {
            Nums = numCums;
            Grad = cumGrad;
        }

        public int Nums { get; set; }

        public NDArray Grad { get; set; }
    }

    public class LBSGD : Optimizer
    {
        public LBSGD(float momentum = 0, bool multi_precision = false, string warmup_strategy = "linear'",
            int warmup_epochs = 5, int batch_scale = 1, int updates_per_epoch = 32, int begin_epoch = 0,
            int num_epochs = 60)
        {
            Logger.Info("Running Large-Batch SGD Algorithm");
            Logger.Info(
                $"(Batch_scale={batch_scale}, warmup_epochs={warmup_epochs}, warmup_strategy={warmup_strategy}, updates_per_epoch={updates_per_epoch})");

            MultiPrecision = multi_precision;
            Momentum = momentum;
            WarmupStrategy = warmup_strategy;
            WarmupEpochs = warmup_epochs;
            BatchScale = batch_scale;
            UpdatesPerEpoch = updates_per_epoch;
            InitUpdates = begin_epoch * updates_per_epoch;
            BeginEpoch = begin_epoch;
            NumEpochs = num_epochs;
            LBMult = 1;
            CumGrads = new Dictionary<int, CumGrad>();
            Adaptive = false;
            ADMult = 1;
        }

        public float Momentum { get; set; }

        public string WarmupStrategy { get; set; }

        public int WarmupEpochs { get; set; }

        public int BatchScale { get; set; }

        public int UpdatesPerEpoch { get; set; }

        public int BeginEpoch { get; set; }

        public int NumEpochs { get; set; }

        public int LBMult { get; set; }

        public Dictionary<int, CumGrad> CumGrads { get; set; }

        public bool Adaptive { get; set; }

        public int ADMult { get; set; }

        public int InitUpdates { get; set; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict();
            state["weight_master_copy"] = null;
            state["momentum"] = null;
            if (MultiPrecision && weight.dtype.Name == DType.Float16.Name)
            {
                state["weight_master_copy"] = weight.AsType(DType.Float32);
                if (Momentum != 0)
                    state["momentum"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);

                return state;
            }

            if (!MultiPrecision && weight.dtype.Name == DType.Float16.Name)
            {
                Logger.Warning("Accumulating with float16 in optimizer can lead to " +
                               "poor accuracy or slow convergence. " +
                               "Consider using multi_precision=True option of the " +
                               "SGD optimizer");
            }

            if (Momentum != 0)
                state["momentum"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);

            return state;
        }

        public override (NDArrayDict, ndarray) CreateStateMultiPrecision(int index, ndarray weight)
        {
            return base.CreateStateMultiPrecision(index, weight);
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotImplementedException();
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            var lr = GetLr(index);
            var wd = GetWd(index);
            UpdateCount(index);

            var cgrad = _cumulate_gradient(grad, index);
            float lbmult = 0;
            if (cgrad.Nums % BatchScale == 0)
            {
                grad = cgrad.Grad / BatchScale;
                if (WarmupStrategy == "lars")
                    lbmult = _get_lars(weight, grad, wd);
                else
                    lbmult = _get_lbmult(cgrad.Nums);

                lr = lr * lbmult;

                var use_multi_precision = state["weight_master_copy"] != null;
                if (!use_multi_precision)
                {
                    if (state["momentum"] != null)
                        weight = nd.SgdMomUpdate(weight, grad, state["momentum"], lr, Momentum, wd, RescaleGrad,
                            ClipGradient.HasValue ? ClipGradient.Value : -1);
                    else
                        weight = nd.SgdUpdate(weight, grad, lr, wd, RescaleGrad,
                            ClipGradient.HasValue ? ClipGradient.Value : -1);
                }
                else
                {
                    if (state["momentum"] != null)
                        weight = nd.MpSgdMomUpdate(weight, grad, state["momentum"], state["weight_master_copy"], lr,
                            Momentum, wd, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1);
                    else
                        weight = nd.MpSgdUpdate(weight, grad, state["weight_master_copy"], lr, wd, RescaleGrad,
                            ClipGradient.HasValue ? ClipGradient.Value : -1);
                }
            }
            else
            {
                lr = 0;
                weight = nd.SgdUpdate(weight, grad, lr, wd, RescaleGrad);
            }
        }

        private float _get_lbmult(float nup)
        {
            var nwup = WarmupEpochs * UpdatesPerEpoch;
            var strategy = WarmupStrategy;
            var maxmult = (float) BatchScale;
            float mult = 0;
            if (nup >= nwup)
            {
                mult = maxmult;
            }
            else if (nwup <= 1)
            {
                mult = 1;
            }
            else
            {
                if (strategy == "linear")
                    mult = 1 + (maxmult - 1) * nup / nwup;
                else if (strategy == "power2")
                    mult = 1 + (maxmult - 1) * (nup * nup) / (nwup * nwup);
                else if (strategy == "sqrt")
                    mult = 1 + (maxmult - 1) * (float) Math.Sqrt(nup / nwup);
                else
                    mult = 1;
            }

            return mult;
        }

        private float _get_lars(NDArray weight, NDArray g, float wd)
        {
            var weight2 = _l2norm(weight);
            var grad2 = _l2norm(g);
            var lars = (float) Math.Sqrt(weight2 / (grad2 + wd * weight2 + 1e-18));
            if (lars < 0.01f)
                lars = 0.01f;
            else if (lars > 100)
                lars = 100;

            return lars;
        }

        private float _l2norm(NDArray v)
        {
            var norm = (v * v).Sum();
            return norm;
        }

        private void _reset_cum_gradient(int index)
        {
            CumGrads[index].Grad = nd.ZerosLike(CumGrads[index].Grad);
        }

        private CumGrad _get_cum_gradient(int index)
        {
            if (CumGrads.ContainsKey(index))
                return CumGrads[index];

            return null;
        }

        private void _put_cum_gradient(int index, CumGrad cgrad)
        {
            CumGrads[index] = cgrad;
        }

        private CumGrad _cumulate_gradient(NDArray grad, int index)
        {
            var cgrad = _get_cum_gradient(index);
            NDArray cum_grad = null;
            var num_cums = 0;
            if (cgrad != null)
            {
                num_cums = cgrad.Nums;
                if (num_cums > 0)
                {
                    cum_grad = cgrad.Grad + grad;
                    num_cums++;
                }
                else
                {
                    cum_grad = grad;
                    num_cums = InitUpdates + 1;
                }
            }
            else
            {
                cum_grad = grad;
                num_cums = InitUpdates + 1;
            }

            cgrad = new CumGrad(num_cums, cum_grad);
            _put_cum_gradient(index, cgrad);

            return cgrad;
        }
    }
}