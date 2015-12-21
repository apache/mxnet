package ml.dmlc.mxnet
import org.slf4j.LoggerFactory

/**
 * Class for learning rate scheduler, which adaptively changes the learning rate
 * based on the training progress.
 * @author Yuan Tang
 */


abstract class LRScheduler(var baseLR: Float = 0.01) {
  /**
   * Base class of a learning rate scheduler
   */

  def apply(numUpdate: Int): Unit
}

class FactorScheduler(var step: Int, var factor: Float) extends LRScheduler {
  /**
   * Class for reducing learning rate in factor
   */

  var count: Int = 0
  private val logger = LoggerFactory.getLogger(classOf[FactorScheduler])

  if (step < 1) {
    throw new IllegalArgumentException("Schedule step must be greater or equal than 1 round")
  }
  if (factor >= 1.0) {
    throw new IllegalArgumentException("Factor must be less than 1 to make lr reduce")
  }

  def apply(numUpdate: Int): Float = {

    if (numUpdate > this.count + this.step) {
      this.count += this.step
      this.baseLR *= this.factor
      this.logger.info(s"""Update$numUpdate: Change learning rate to ${this.baseLR}%.5f""")
    }
    this.baseLR
  }
}
