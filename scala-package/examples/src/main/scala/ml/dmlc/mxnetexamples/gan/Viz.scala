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

package ml.dmlc.mxnetexamples.gan

import org.opencv.core.Core
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import ml.dmlc.mxnet.NDArray
import org.opencv.core.Mat
import org.opencv.core.CvType
import java.util.ArrayList
import org.opencv.core.Size

object Viz {

  nu.pattern.OpenCV.loadShared()

  private def clip(x: Array[Float]): Array[Byte] = {
    x.map(_ * 255f).map(x => if (x < 0f) 0 else if (x > 255f) 255 else x.toInt)
      .map(_.toByte)
  }

  private def getImg(rawData: Array[Byte],
    channels: Int, height: Int, width: Int, flip: Boolean): Mat = {
    val totals = height * width
     val img = if (channels > 1) { // rbg image
      val (rA, gA, bA) = {
        val tmp = rawData.grouped(totals).toArray
        (tmp(0), tmp(1), tmp(2))
      }

      val rr = new Mat(height, width, CvType.CV_8U)
      rr.put(0, 0, rA)
      val gg = new Mat(height, width, CvType.CV_8U)
      gg.put(0, 0, gA)
      val bb = new Mat(height, width, CvType.CV_8U)
      bb.put(0, 0, bA)

      val result = new Mat()
      val layers = new ArrayList[Mat]()
      layers.add(bb)
      layers.add(gg)
      layers.add(rr)
      Core.merge(layers, result)
      result
    } else { // gray image
      val result = new Mat(height, width, CvType.CV_8U)
      result.put(0, 0, rawData)
      result
    }
    if (flip) {
      val result = new Mat()
      Core.flip(img, result, 0)
      result
    } else img
  }

  def imSave(title: String, outputPath: String, x: NDArray, flip: Boolean = false): Unit = {
    val shape = x.shape
    assert(shape.length == 4)

    val (n, c, h, w) = (shape(0), shape(1), shape(2), shape(3))

    val totals = h * w
    val rawData = clip(x.toArray)

    val img = {
      val row, col = Math.sqrt(n).toInt
      val lineArrs = rawData.grouped(col * c * totals)

      val lineMats = new ArrayList[Mat]()

      for (line <- lineArrs) {
        val imgArr = line.grouped(c * totals)
        val colMats = new Mat
        val src = new ArrayList[Mat]()

        for(arr <- imgArr) src.add(getImg(arr, c, h, w, flip))

        Core.hconcat(src, colMats)
        lineMats.add(colMats)
      }
      val result = new Mat()
      Core.vconcat(lineMats, result)
      result
    }
    val resizedImg = new Mat
    Imgproc.resize(img, resizedImg, new Size(img.width() * 1.5, img.height() * 1.5))
    Highgui.imwrite(s"$outputPath/$title.jpg", resizedImg)
  }
}
