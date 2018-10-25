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

package org.apache.mxnet

import java.net.URL

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * Image API of Scala package
  * enable OpenCV feature
  */
object Image {

  def imDecode (urlStr : String) : NDArrayFuncReturn = {
    val url = new URL(urlStr)
    val inputStream = url.openStream
    val buffer = new Array[Byte](2048)
    val arrBuffer = ArrayBuffer[Byte]()
    var length = 0
    while (length != -1) {
      length = inputStream.read(buffer)
      if (length != -1) arrBuffer ++= buffer.slice(0, length)
    }
    imDecode(arrBuffer.toArray)
  }
  /**
    * Decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param buf    Buffer containing binary encoded image
    * @param flag   Convert decoded image to grayscale (0) or color (1).
    * @param to_rgb Whether to convert decoded image
    *               to mxnet's default RGB format (instead of opencv's default BGR).
    * @return org.apache.mxnet.NDArray
    */
  def imDecode (buf : Array[Byte], flag : Option[Int] = None,
                to_rgb : Option[Boolean] = None,
                out : Option[NDArray] = None) : org.apache.mxnet.NDArrayFuncReturn = {
    val nd = NDArray.array(buf.map(_.toFloat), Shape(buf.length))
    val byteND = NDArray.api.cast(nd, "uint8")
    val args : ListBuffer[Any] = ListBuffer()
    val map : mutable.Map[String, Any] = mutable.Map()
    args += byteND
    if (flag.isDefined) map("flag") = flag.get
    if (to_rgb.isDefined) map("to_rgb") = to_rgb.get
    if (out.isDefined) map("out") = out.get
    NDArray.genericNDArrayFunctionInvoke("_cvimdecode", args, map.toMap)
  }

  /**
    * Read and decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param filename Name of the image file to be loaded.
    * @param flag     Convert decoded image to grayscale (0) or color (1).
    * @param to_rgb   Whether to convert decoded image to mxnet's default RGB format
    *                 (instead of opencv's default BGR).
    * @return org.apache.mxnet.NDArray in HWC format
    */
  def imRead (filename : String, flag : Option[Int] = None,
                 to_rgb : Option[Boolean] = None,
                 out : Option[NDArray] = None) : org.apache.mxnet.NDArrayFuncReturn = {
    val args : ListBuffer[Any] = ListBuffer()
    val map : mutable.Map[String, Any] = mutable.Map()
    map("filename") = filename
    if (flag.isDefined) map("flag") = flag.get
    if (to_rgb.isDefined) map("to_rgb") = to_rgb.get
    if (out.isDefined) map("out") = out.get
    NDArray.genericNDArrayFunctionInvoke("_cvimread", args, map.toMap)
  }

  /**
    * Resize image with OpenCV.
    * @param src     source image in NDArray
    * @param w       Width of resized image.
    * @param h       Height of resized image.
    * @param interp  Interpolation method (default=cv2.INTER_LINEAR).
    * @return org.apache.mxnet.NDArray
    */
  def imResize (src : org.apache.mxnet.NDArray, w : Int, h : Int,
                   interp : Option[Int] = None,
                   out : Option[NDArray] = None) : org.apache.mxnet.NDArrayFuncReturn = {
    val args : ListBuffer[Any] = ListBuffer()
    val map : mutable.Map[String, Any] = mutable.Map()
    args += src
    map("w") = w
    map("h") = h
    if (interp.isDefined) map("interp") = interp.get
    if (out.isDefined) map("out") = out.get
    NDArray.genericNDArrayFunctionInvoke("_cvimresize", args, map.toMap)
  }

  /**
    * Pad image border with OpenCV.
    * @param src    source image
    * @param top    Top margin.
    * @param bot    Bottom margin.
    * @param left   Left margin.
    * @param right  Right margin.
    * @param typeOf Filling type (default=cv2.BORDER_CONSTANT).
    * @param value  (Deprecated! Use ``values`` instead.) Fill with single value.
    * @param values Fill with value(RGB[A] or gray), up to 4 channels.
    * @return org.apache.mxnet.NDArray
    */
  def copyMakeBorder (src : org.apache.mxnet.NDArray, top : Int, bot : Int,
                      left : Int, right : Int, typeOf : Option[Int] = None,
                      value : Option[Double] = None, values : Option[Any] = None,
                      out : Option[NDArray] = None) : org.apache.mxnet.NDArrayFuncReturn = {
    val args : ListBuffer[Any] = ListBuffer()
    val map : mutable.Map[String, Any] = mutable.Map()
    args += src
    map("top") = top
    map("bot") = bot
    map("left") = left
    map("right") = right
    if (typeOf.isDefined) map("type") = typeOf.get
    if (value.isDefined) map("value") = value.get
    if (values.isDefined) map("values") = values.get
    if (out.isDefined) map("out") = out.get
    NDArray.genericNDArrayFunctionInvoke("_cvcopyMakeBorder", args, map.toMap)
  }

}
