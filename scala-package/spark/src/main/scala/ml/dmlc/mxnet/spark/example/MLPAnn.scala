package ml.dmlc.mxnet.spark.example

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.{Logger, LoggerFactory}

object MLPAnn {
  private val logger: Logger = LoggerFactory.getLogger(classOf[MLPAnn])
  //private var numWorker: Int = 1

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkMLP")
    val sc = new SparkContext(conf)

    val trainFile = args(0)
    val valFile = args(1)

    val trainRaw = sc.textFile(trainFile)
    val data = trainRaw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }

    val valRaw = sc.textFile(valFile)
    val valData = valRaw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }

    val sqlContext = new SQLContext(sc)
    val df = sqlContext.createDataFrame(data, classOf[LabeledPoint])
    // tricky persist to avoid additional cache
    val layers = Array[Int](784, 128, 64, 10)
    // create the trainer and set its parameters
    val start = System.currentTimeMillis
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setTol(1e-10)
      .setSeed(1234L)
      .setMaxIter(10)
    // train the model
    val model = trainer.fit(df)
    val timeCost = System.currentTimeMillis - start
    logger.info("Training cost {} milli seconds", timeCost)

    // compute precision on the test set
    val dfVal = sqlContext.createDataFrame(valData, classOf[LabeledPoint])
    val result = model.transform(dfVal)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    logger.info("Precision:" + evaluator.evaluate(predictionAndLabels))

    sc.stop()
  }
}

class MLPAnn
