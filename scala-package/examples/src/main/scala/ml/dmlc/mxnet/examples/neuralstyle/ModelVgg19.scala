package ml.dmlc.mxnet.examples.neuralstyle

import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape

/**
 * Definition for the neuralstyle network and initialize it with pretrained weight
 * @author Depeng Liang
 */
object ModelVgg19 {
  case class ConvExecutor(executor: Executor, data: NDArray, dataGrad: NDArray,
                      style: Array[NDArray], content: NDArray, argDict: Map[String, NDArray])

  def getSymbol(): (Symbol, Symbol) = {
    // declare symbol
    val data = Symbol.Variable("data")
    val conv1_1 = Symbol.Convolution("conv1_1")(Map("data" -> data , "num_filter" -> 64,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu1_1 = Symbol.Activation("relu1_1")(Map("data" -> conv1_1 , "act_type" -> "relu"))
    val conv1_2 = Symbol.Convolution("conv1_2")(Map("data" -> relu1_1 , "num_filter" -> 64,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu1_2 = Symbol.Activation("relu1_2")(Map("data" -> conv1_2 , "act_type" -> "relu"))
    val pool1 = Symbol.Pooling("pool1")(Map("data" -> relu1_2 , "pad" -> "(0,0)",
                                    "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv2_1 = Symbol.Convolution("conv2_1")(Map("data" -> pool1 , "num_filter" -> 128,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu2_1 = Symbol.Activation("relu2_1")(Map("data" -> conv2_1 , "act_type" -> "relu"))
    val conv2_2 = Symbol.Convolution("conv2_2")(Map("data" -> relu2_1 , "num_filter" -> 128,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu2_2 = Symbol.Activation("relu2_2")(Map("data" -> conv2_2 , "act_type" -> "relu"))
    val pool2 = Symbol.Pooling("pool2")(Map("data" -> relu2_2 , "pad" -> "(0,0)",
                                    "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv3_1 = Symbol.Convolution("conv3_1")(Map("data" -> pool2 , "num_filter" -> 256,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu3_1 = Symbol.Activation("relu3_1")(Map("data" -> conv3_1 , "act_type" -> "relu"))
    val conv3_2 = Symbol.Convolution("conv3_2")(Map("data" -> relu3_1 , "num_filter" -> 256,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu3_2 = Symbol.Activation("'relu3_2")(Map("data" -> conv3_2 , "act_type" -> "relu"))
    val conv3_3 = Symbol.Convolution("conv3_3")(Map("data" -> relu3_2 , "num_filter" -> 256,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu3_3 = Symbol.Activation("relu3_3")(Map("data" -> conv3_3 , "act_type" -> "relu"))
    val conv3_4 = Symbol.Convolution("conv3_4")(Map("data" -> relu3_3 , "num_filter" -> 256,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu3_4 = Symbol.Activation("relu3_4")(Map("data" -> conv3_4 , "act_type" -> "relu"))
    val pool3 = Symbol.Pooling("pool3")(Map("data" -> relu3_4 , "pad" -> "(0,0)",
                                    "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv4_1 = Symbol.Convolution("conv4_1")(Map("data" -> pool3 , "num_filter" -> 512,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu4_1 = Symbol.Activation("relu4_1")(Map("data" -> conv4_1 , "act_type" -> "relu"))
    val conv4_2 = Symbol.Convolution("conv4_2")(Map("data" -> relu4_1 , "num_filter" -> 512,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu4_2 = Symbol.Activation("relu4_2")(Map("data" -> conv4_2 , "act_type" -> "relu"))
    val conv4_3 = Symbol.Convolution("conv4_3")(Map("data" -> relu4_2 , "num_filter" -> 512,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu4_3 = Symbol.Activation("relu4_3")(Map("data" -> conv4_3 , "act_type" -> "relu"))
    val conv4_4 = Symbol.Convolution("conv4_4")(Map("data" -> relu4_3 , "num_filter" -> 512,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu4_4 = Symbol.Activation("relu4_4")(Map("data" -> conv4_4 , "act_type" -> "relu"))
    val pool4 = Symbol.Pooling("pool4")(Map("data" -> relu4_4 , "pad" -> "(0,0)",
                                    "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv5_1 = Symbol.Convolution("conv5_1")(Map("data" -> pool4 , "num_filter" -> 512,
                                        "pad" -> "(1,1)", "kernel" -> "(3,3)", "stride" -> "(1,1)",
                                        "no_bias" -> false, "workspace" -> 1024))
    val relu5_1 = Symbol.Activation("relu5_1")(Map("data" -> conv5_1 , "act_type" -> "relu"))

    // style and content layers
    val style = Symbol.Group(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
    val content = Symbol.Group(relu4_2)
    (style, content)
  }

  def getExecutor(style: Symbol, content: Symbol, modelPath: String,
      inputSize: (Int, Int), ctx: Context): ConvExecutor = {
    val out = Symbol.Group(style, content)
    // make executor
    val (argShapes, outputShapes, auxShapes) = out.inferShape(
      Map("data" -> Shape(1, 3, inputSize._1, inputSize._2)))
    val argNames = out.listArguments()
    val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
    val gradDict = Map("data" -> argDict("data").copyTo(ctx))
    // init with pretrained weight
    val pretrained = NDArray.load2Map(modelPath)
    argNames.filter(_ != "data").foreach { name =>
      val key = s"arg:$name"
      if (pretrained.contains(key)) argDict(name).set(pretrained(key))
    }
    val executor = out.bind(ctx, argDict, gradDict)
    val outArray = executor.outputs
    ConvExecutor(executor = executor,
                              data = argDict("data"),
                              dataGrad = gradDict("data"),
                              style = outArray.take(outArray.length - 1),
                              content = outArray(outArray.length - 1),
                              argDict = argDict)
    }

  def getModel(modelPath: String, inputSize: (Int, Int), ctx: Context): ConvExecutor = {
    val (style, content) = getSymbol()
    getExecutor(style, content, modelPath, inputSize, ctx)
  }
}
