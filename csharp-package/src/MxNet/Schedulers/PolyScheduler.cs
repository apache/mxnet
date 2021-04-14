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

namespace MxNet.Schedulers
{
    public class PolyScheduler : LRScheduler
    {
        public PolyScheduler(int max_update, float base_lr = 0.01F, int pwr = 2, float final_lr = 0,
            int warmup_steps = 0,
            float warmup_begin_lr = 0, string warmup_mode = "linear")
            : base(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        {
            if (max_update < 1)
                throw new ArgumentException("maximum number of updates must be strictly positive");
            MaxUpdate = max_update;
            Power = pwr;
            FinalLr = final_lr;
            BaseLrOrig = base_lr;
            MaxSteps = max_update - warmup_steps;
        }

        public int MaxUpdate { get; }
        public int Power { get; }
        public float FinalLr { get; }
        public float BaseLrOrig { get; }
        public int MaxSteps { get; }

        public override float Call(uint num_update)
        {
            if (num_update < WarmupSteps)
                return GetWarmupLR(num_update);

            if (num_update <= MaxUpdate)
                BaseLearningRate = FinalLr + (BaseLrOrig - FinalLr) *
                    (float) Math.Pow(1 - (num_update - WarmupSteps) / MaxSteps, Power);

            return BaseLearningRate;
        }
    }
}