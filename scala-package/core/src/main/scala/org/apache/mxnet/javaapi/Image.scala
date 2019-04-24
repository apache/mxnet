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

package org.apache.mxnet.javaapi
// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on
import java.io.InputStream
import scala.collection.JavaConverters._

object Image {
  /**
    * Decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param buf   Buffer containing binary encoded image
    * @param flag  Convert decoded image to grayscale (0) or color (1).
    * @param toRGB Whether to convert decoded image
    *              to mxnet's default RGB format (instead of opencv's default BGR).
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(buf: Array[Byte], flag: Int, toRGB: Boolean): NDArray = {
    org.apache.mxnet.Image.imDecode(buf, flag, toRGB, None)
  }

  /**
    * Decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param buf   Buffer containing binary encoded image
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(buf: Array[Byte]): NDArray = {
    imDecode(buf, 1, true)
  }

  /**
    * Same imageDecode with InputStream
    *
    * @param inputStream the inputStream of the image
    * @param flag        Convert decoded image to grayscale (0) or color (1).
    * @param toRGB       Whether to convert decoded image
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(inputStream: InputStream, flag: Int, toRGB: Boolean): NDArray = {
    org.apache.mxnet.Image.imDecode(inputStream, flag, toRGB, None)
  }

  /**
    * Same imageDecode with InputStream
    *
    * @param inputStream the inputStream of the image
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(inputStream: InputStream): NDArray = {
    imDecode(inputStream, 1, true)
  }

  /**
    * Read and decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param filename Name of the image file to be loaded.
    * @param flag     Convert decoded image to grayscale (0) or color (1).
    * @param toRGB    Whether to convert decoded image to mxnet's default RGB format
    *                 (instead of opencv's default BGR).
    * @return org.apache.mxnet.NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imRead(filename: String, flag: Int, toRGB: Boolean): NDArray = {
    org.apache.mxnet.Image.imRead(filename, Some(flag), Some(toRGB), None)
  }

  /**
    * Read and decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param filename Name of the image file to be loaded.
    * @return org.apache.mxnet.NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imRead(filename: String): NDArray = {
    imRead(filename, 1, true)
  }

  /**
    * Resize image with OpenCV.
    * @param src    source image in NDArray
    * @param w      Width of resized image.
    * @param h      Height of resized image.
    * @param interp Interpolation method (default=cv2.INTER_LINEAR).
    * @return org.apache.mxnet.NDArray
    */
  def imResize(src: NDArray, w: Int, h: Int, interp: Integer): NDArray = {
    val interpVal = if (interp == null) None else Some(interp.intValue())
    org.apache.mxnet.Image.imResize(src, w, h, interpVal, None)
  }

  /**
    * Resize image with OpenCV.
    * @param src    source image in NDArray
    * @param w      Width of resized image.
    * @param h      Height of resized image.
    * @return org.apache.mxnet.NDArray
    */
  def imResize(src: NDArray, w: Int, h: Int): NDArray = {
    imResize(src, w, h, null)
  }

  /**
    * Do a fixed crop on the image
    * @param src Src image in NDArray
    * @param x0  starting x point
    * @param y0  starting y point
    * @param w   width of the image
    * @param h   height of the image
    * @return cropped NDArray
    */
  def fixedCrop(src: NDArray, x0: Int, y0: Int, w: Int, h: Int): NDArray = {
    org.apache.mxnet.Image.fixedCrop(src, x0, y0, w, h)
  }

  /**
    * Convert a NDArray image to a real image
    * The time cost will increase if the image resolution is big
    * @param src Source image file in RGB
    * @return Buffered Image
    */
  def toImage(src: NDArray): BufferedImage = {
    org.apache.mxnet.Image.toImage(src)
  }

  /**
    * Draw bounding boxes on the image
    * @param src        buffered image to draw on
    * @param coordinate Contains Map of xmin, xmax, ymin, ymax
    *                   corresponding to top-left and down-right points
    * @param names      The name set of the bounding box
    */
  def drawBoundingBox(src: BufferedImage,
                      coordinate: java.util.List[
                        java.util.Map[java.lang.String, java.lang.Integer]],
                      names: java.util.List[java.lang.String]): Unit = {
    val coord = coordinate.asScala.map(
      _.asScala.map{case (name, value) => (name, Integer2int(value))}.toMap).toArray
    org.apache.mxnet.Image.drawBoundingBox(src, coord, Option(names.asScala.toArray))
  }

}
