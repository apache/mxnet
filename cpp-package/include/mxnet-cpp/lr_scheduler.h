/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
*  Copyright (c) 2017 by Contributors
* \file lr_scheduler.h
* \brief Scheduling learning rate
*/

#ifndef MXNET_CPP_LR_SCHEDULER_H_
#define MXNET_CPP_LR_SCHEDULER_H_

#include "dmlc/logging.h"

namespace mxnet {
namespace cpp {

/*!
* \brief lr scheduler interface
*/
class LRScheduler {
 public:
  /*!
  * \brief constructor
  * \param base_lr the initial learning rate.
  */
  explicit LRScheduler(float base_lr = 0.01)
      : base_lr_(base_lr) {}
  /*!
  * \brief set base lr
  * \param lr learning rate from optimizer
  */
  void SetLR(const float lr) { base_lr_ = lr; }
  /*!
  * \brief get a new learning rate
  */
  virtual float GetLR(unsigned num_update) = 0;
  /*!
  * \brief destructor
  */
  virtual ~LRScheduler() {}

 protected:
  float base_lr_;
};

class FactorScheduler : public LRScheduler {
 public:
  explicit FactorScheduler(int step, float factor = 1, float stop_factor_lr = 1e-8)
      : LRScheduler() {
    step_ = step;
    factor_ = factor;
    stop_factor_lr_ = stop_factor_lr;
  }

  float GetLR(unsigned num_update) override {
    while (num_update > unsigned(count_ + step_)) {
      count_ += step_;
      base_lr_ *= factor_;
      if (base_lr_ < stop_factor_lr_) {
        base_lr_ = stop_factor_lr_;
        LG << "Update[" << num_update << "]: now learning rate arrived at " \
           << base_lr_ << ", will not change in the future";
      } else {
        LG << "Update[" << num_update << "]: Change learning rate to " << base_lr_;
      }
    }
    return base_lr_;
  }

 private:
  int count_ = 0;
  int step_;
  float factor_;
  float stop_factor_lr_;
};

class PolyScheduler : public LRScheduler {
 public:
    explicit PolyScheduler(unsigned warmup_steps, unsigned max_update, float power = 2.f,
        float final_lr = 0)
            : LRScheduler(), warmup_steps_(warmup_steps), max_update_(max_update),
            max_udpate_(static_cast<float>(max_update)), power_(power), final_lr_(final_lr) {}

    float GetLR(unsigned num_update) override {
      if (num_update <= max_update_) {
        current_lr_ = final_lr_ + (base_lr_ - final_lr_) *
          powf((1.f - static_cast<float>(num_update - warmup_steps_)/max_update_), power_);
        LG << "Update[" << num_update << "]: Learning rate has arrived at "
         << final_lr_ << "\n";
      }
      return current_lr_;
    }

 private:
  unsigned warmup_steps_;
  unsigned max_update_;
  float max_udpate_;
  float power_;
  float final_lr_;
  float current_lr_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_LR_SCHEDULER_H_
