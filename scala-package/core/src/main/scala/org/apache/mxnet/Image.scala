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
// scalastyle:off
import java.awt.{BasicStroke, Color, Graphics2D}
import java.awt.image.BufferedImage
// scalastyle:on
import java.io.InputStream

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * Image API of Scala package
  * enable OpenCV feature
  */
object Image {

  /**
    * Decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param buf    Buffer containing binary encoded image
    * @param flag   Convert decoded image to grayscale (0) or color (1).
    * @param to_rgb Whether to convert decoded image
    *               to mxnet's default RGB format (instead of opencv's default BGR).
    * @param out    NDArray to store the output
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(buf: Array[Byte], flag: Int,
               to_rgb: Boolean,
               out: Option[NDArray]): NDArray = {
    val nd = NDArray.array(buf.map( x => (x & 0xFF).toFloat), Shape(buf.length))
    val byteND = NDArray.api.cast(nd, "uint8")
    val args : ListBuffer[Any] = ListBuffer()
    val map : mutable.Map[String, Any] = mutable.Map()
    args += byteND
    map("flag") = flag
    map("to_rgb") = to_rgb
    if (out.isDefined) map("out") = out.get
    NDArray.genericNDArrayFunctionInvoke("_cvimdecode", args, map.toMap)
  }

  /**
    * Same imageDecode with InputStream
    * @param inputStream the inputStream of the image
    * @param flag   Convert decoded image to grayscale (0) or color (1).
    * @param to_rgb Whether to convert decoded image
    *               to mxnet's default RGB format (instead of opencv's default BGR).
    * @param out    NDArray to store the output
    * @return NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imDecode(inputStream: InputStream, flag: Int = 1,
               to_rgb: Boolean = true,
               out: Option[NDArray] = None): NDArray = {
    val buffer = new Array[Byte](2048)
    val arrBuffer = ArrayBuffer[Byte]()
    var length = 0
    while (length != -1) {
      length = inputStream.read(buffer)
      if (length != -1) arrBuffer ++= buffer.slice(0, length)
    }
    imDecode(arrBuffer.toArray, flag, to_rgb, out)
  }

  /**
    * Read and decode image with OpenCV.
    * Note: return image in RGB by default, instead of OpenCV's default BGR.
    * @param filename Name of the image file to be loaded.
    * @param flag     Convert decoded image to grayscale (0) or color (1).
    * @param to_rgb   Whether to convert decoded image to mxnet's default RGB format
    *                 (instead of opencv's default BGR).
    * @param out    NDArray to store the output
    * @return org.apache.mxnet.NDArray in HWC format with DType [[DType.UInt8]]
    */
  def imRead(filename: String, flag: Option[Int] = None,
             to_rgb: Option[Boolean] = None,
             out: Option[NDArray] = None): NDArray = {
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
    * @param out    NDArray to store the output
    * @return org.apache.mxnet.NDArray
    */
  def imResize(src: org.apache.mxnet.NDArray, w: Int, h: Int,
               interp: Option[Int] = None,
               out: Option[NDArray] = None): NDArray = {
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
    * @param out    NDArray to store the output
    * @return org.apache.mxnet.NDArray
    */
  def copyMakeBorder(src: org.apache.mxnet.NDArray, top: Int, bot: Int,
                     left: Int, right: Int, typeOf: Option[Int] = None,
                     value: Option[Double] = None, values: Option[Any] = None,
                     out: Option[NDArray] = None): NDArray = {
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

  /**
    * Do a fixed crop on the image
    * @param src Src image in NDArray
    * @param x0 starting x point
    * @param y0 starting y point
    * @param w width of the image
    * @param h height of the image
    * @return cropped NDArray
    */
  def fixedCrop(src: NDArray, x0: Int, y0: Int, w: Int, h: Int): NDArray = {
    NDArray.api.crop(src, Shape(y0, x0, 0), Shape(y0 + h, x0 + w, src.shape.get(2)))
  }

  /**
    * Convert a NDArray image to a real image
    * The time cost will increase if the image resolution is big
    * @param src Source image file in RGB
    * @return Buffered Image
    */
  def toImage(src: NDArray): BufferedImage = {
    require(src.dtype == DType.UInt8, "The input NDArray must be bytes")
    require(src.shape.length == 3, "The input should contains height, width and channel")
    require(src.shape(2) == 3, "There should be three channels: RGB")
    val height = src.shape.get(0)
    val width = src.shape.get(1)
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val arr = src.toArray
    (0 until height).par.foreach(r => {
      (0 until width).par.foreach(c => {
        // NDArray in RGB
        val cellIndex = r * width * 3 + c * 3
        val red = arr(cellIndex).toByte & 0xFF
        val green = arr(cellIndex + 1).toByte & 0xFF
        val blue = arr(cellIndex + 2).toByte & 0xFF
        val rgb = (red << 16) | (green << 8) | blue
        img.setRGB(c, r, rgb)
      })
    })
    img
  }

  /**
    * Helper function to generate ramdom colors
    * @param transparency The transparency level
    * @return Color
    */
  private def randomColor(transparency: Option[Float] = Some(1.0f)) : Color = {
    new Color(
      Math.random().toFloat, Math.random().toFloat, Math.random().toFloat,
      transparency.get
    )
  }

  /**
    * Method to draw bounding boxes for an image
    * @param src Source of the buffered image
    * @param coordinate Contains Map of xmin, xmax, ymin, ymax
    *                   corresponding to top-left and down-right points
    * @param names The name set of the bounding box
    * @param stroke Thickness of the bounding box
    * @param fontSizeMult Font size multiplier
    * @param transparency Transparency of the bounding box
    */
  def drawBoundingBox(src: BufferedImage, coordinate: Array[Map[String, Int]],
                      names: Option[Array[String]] = None,
                      stroke : Option[Int] = Some(3),
                      fontSizeMult : Option[Float] = Some(1.0f),
                      transparency: Option[Float] = Some(1.0f)): Unit = {
    val g2d : Graphics2D = src.createGraphics()
    g2d.setStroke(new BasicStroke(stroke.get))
    // Increase the size of font
    val currentFont = g2d.getFont
    val newFont = currentFont.deriveFont(currentFont.getSize * fontSizeMult.get)
    g2d.setFont(newFont)
    // Get font metrics to draw the font box
    val fm = g2d.getFontMetrics(newFont)
    for (idx <- coordinate.indices) {
      val map = coordinate(idx)
      g2d.setColor(randomColor(transparency).darker())
      g2d.drawRect(map("xmin"), map("ymin"), map("xmax") - map("xmin"), map("ymax") - map("ymin"))
      // Write the name of the bounding box
      if (names.isDefined) {
        val x = map("xmin") - stroke.get
        val y = map("ymin")
        val h = fm.getHeight
        val w = fm.charsWidth(names.get(idx).toCharArray, 0, names.get(idx).length())
        g2d.fillRect(x, y - h, w, h)
        g2d.setColor(Color.WHITE)
        g2d.drawString(names.get(idx), x, y)
      }
    }
    g2d.dispose()
  }

}
