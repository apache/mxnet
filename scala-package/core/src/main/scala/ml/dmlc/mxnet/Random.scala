package ml.dmlc.mxnet


import ml.dmlc.mxnet.NDArray.{_randomUniform}
/**
 * Created by yuantang on 12/27/15.
 */
class Random {

  def uniform(low: Float, high: Float, shape: Array[Int]=null, ctx: Context=null, out: NDArray=null): NDArray = {
    if (out != null) {
      require(shape == null & ctx == null, "shape and ctx is not needed when out is specified.")
    } else {
      require(shape != null, "shape is required when out is not specified")
      var out = NDArray.empty(shape, ctx)
    }
    return NDArray._randomUniform(low, high, out)
  }

  def normal(mean: Float, stdvar: Float, shape: Array[Int]=null, ctx: Context=null, out: NDArray=null): NDArray = {
    if (out != null) {
      require(shape == null & ctx == null, "shape and ctx is not needed when out is specified.")
    } else {
      require(shape != null, "shape is required when out is not specified")
      var out = NDArray.empty(shape, ctx)
    }
    return NDArray._randomGaussian(mean, stdvar, out)
  }
}
