package org.apache.mxnet.infer.javaapi

class ObjectDetectorOutput (className: String, args: Array[Float]){

  private val probability = args(0)
  private val xMin = args(1)
  private val xMax = args(2)
  private val yMin = args(3)
  private val yMax = args(4)

  def getClassName = className

  def getProbability = probability

  def getXMin = xMin

  def getXMax = xMax

  def getYMin = yMin

  def getYMax = yMax

}
