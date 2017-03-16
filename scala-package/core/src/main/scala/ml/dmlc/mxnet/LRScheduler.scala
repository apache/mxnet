/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.dmlc.mxnet

import org.slf4j.LoggerFactory

/**
 * Learning rate scheduler, which adaptively changes the learning rate
 * based on the training progress.
 * @author Yuan Tang
 */

abstract class LRScheduler(var baseLR: Float = 0.01f) {
  /**
   * Base class of a learning rate scheduler
   *
   * The training progress is presented by `num_update`, which can be roughly
   * viewed as the number of minibatches executed so far. Its value is
   * non-decreasing, and increases at most by one.
   *
   * The exact value is the upper bound of the number of updates applied to
   * a weight/index.
   *
   * @param numUpdate Int, the maximal number of updates applied to a weight.
   */
  def apply(numUpdate: Int): Float
}

/**
 * Class for reducing learning rate in factor
 *
 * Assume the weight has been updated by n times, then the learning rate will
 * be base_lr * factor^^(floor(n/step))
 *
 * @param step Int, schedule learning rate after n updates
 * @param factor Float, the factor for reducing the learning rate
 *
 */
class FactorScheduler(protected var step: Int, protected var factor: Float) extends LRScheduler {

  protected var count: Int = 0
  private val logger = LoggerFactory.getLogger(classOf[FactorScheduler])

  require(step >= 1, "Schedule step must be greater or equal than 1 round")
  require(factor < 1.0, "Factor must be less than 1 to make lr reduce")

  def apply(numUpdate: Int): Float = {

    if (numUpdate > this.count + this.step) {
      this.count += this.step
      this.baseLR *= this.factor
      this.logger.info(s"Update$numUpdate: Change learning rate to ${this.baseLR}")
    }
    this.baseLR
  }
}
