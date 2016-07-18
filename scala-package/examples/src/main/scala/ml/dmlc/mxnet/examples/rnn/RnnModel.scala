package ml.dmlc.mxnet.examples.rnn

import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Symbol

object RnnModel {
  class LSTMInferenceModel(numLstmLayer: Int, inputSize: Int, numHidden: Int,
                           numEmbed: Int, numLabel: Int, argParams: Map[String, NDArray],
                           ctx: Context = Context.cpu(), dropout: Float = 0f) {
    private val sym = Lstm.lstmInferenceSymbol(numLstmLayer,
                                               inputSize,
                                               numHidden,
                                               numEmbed,
                                               numLabel,
                                               dropout)
    private val batchSize = 1
    private val initC = (for (l <- 0 until numLstmLayer)
                          yield (s"l${l}_init_c" -> Shape(batchSize, numHidden))).toMap
    private val initH = (for (l <- 0 until numLstmLayer)
                          yield (s"l${l}_init_h" -> Shape(batchSize, numHidden))).toMap
    private val dataShape = Map("data" -> Shape(batchSize))
    private val inputShape = initC ++ initH ++ dataShape
    private val executor = sym.simpleBind(ctx = ctx, shapeDict = inputShape)

    for (key <- this.executor.argDict.keys) {
      if (!inputShape.contains(key) && argParams.contains(key) && key != "softmax_label") {
        argParams(key).copyTo(this.executor.argDict(key))
      }
    }

    private var stateName = (Array[String]() /: (0 until numLstmLayer)) { (acc, i) =>
      acc :+ s"l${i}_init_c"  :+ s"l${i}_init_h"
    }

    private val statesDict = stateName.zip(this.executor.outputs.drop(1)).toMap
    private val inputArr = NDArray.zeros(dataShape("data"))

    def forward(inputData: NDArray, newSeq: Boolean = false): Array[Float] = {
      if (newSeq == true) {
        for (key <- this.statesDict.keys) {
          this.executor.argDict(key).set(0f)
        }
      }
      inputData.copyTo(this.executor.argDict("data"))
      this.executor.forward()
      for (key <- this.statesDict.keys) {
        this.statesDict(key).copyTo(this.executor.argDict(key))
      }
      val prob = this.executor.outputs(0).toArray
      prob
    }
  }
}
