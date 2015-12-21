package ml.dmlc.mxnet
import org.slf4j.LoggerFactory

/**
 * Learning rate scheduler, which adaptively changes the learning rate
 * based on the training progress.
 * @author Yuan Tang
 */

abstract class LRScheduler(var baseLR: Double = 0.01) {
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
  def apply(numUpdate: Int): Double
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
class FactorScheduler(var step: Int, var factor: Float) extends LRScheduler {

  var count: Int = 0
  private val logger = LoggerFactory.getLogger(classOf[FactorScheduler])

  if (step < 1) {
    throw new IllegalArgumentException("Schedule step must be greater or equal than 1 round")
  }
  if (factor >= 1.0) {
    throw new IllegalArgumentException("Factor must be less than 1 to make lr reduce")
  }

  def apply(numUpdate: Int): Double = {

    if (numUpdate > this.count + this.step) {
      this.count += this.step
      this.baseLR *= this.factor
      this.logger.info(s"""Update$numUpdate: Change learning rate to ${this.baseLR}""")
    }
    this.baseLR
  }
}

