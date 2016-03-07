package ml.dmlc.mxnet

import org.slf4j.{Logger, LoggerFactory}

/**
 * Callback functions that can be used to track various status during epoch.
 * @author Yizhi Liu
 */
object Callback {
  class Speedometer(val batchSize: Int, val frequent: Int = 50) extends BatchEndCallback {
    private val logger: Logger = LoggerFactory.getLogger(classOf[Speedometer])
    private var init = false
    private var tic: Long = 0L
    private var lastCount: Int = 0

    override def invoke(epoch: Int, count: Int, evalMetric: EvalMetric): Unit = {
      if (lastCount > count) {
        init = false
      }
      lastCount = count

      if (init) {
        if (count % frequent == 0) {
          val speed = frequent.toDouble * batchSize / (System.currentTimeMillis - tic) * 1000
          if (evalMetric != null) {
            val (name, value) = evalMetric.get
            logger.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f".format(
              epoch, count, speed, name, value))
          } else {
            logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec".format(epoch, count, speed))
          }
          tic = System.currentTimeMillis
        }
      } else {
        init = true
        tic = System.currentTimeMillis
      }
    }
  }
}
