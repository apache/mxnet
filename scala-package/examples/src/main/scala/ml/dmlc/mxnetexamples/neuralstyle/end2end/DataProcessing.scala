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

package ml.dmlc.mxnetexamples.neuralstyle.end2end

import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.Pixel
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.JpegWriter
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.NDArray
import java.io.File
import ml.dmlc.mxnet.Shape
import scala.util.Random

/**
 * @author Depeng Liang
 */
object DataProcessing {

  def preprocessContentImage(path: String,
      dShape: Shape = null, ctx: Context): NDArray = {
    val img = Image(new File(path))
    val resizedImg = img.scaleTo(dShape(3), dShape(2))
    val sample = NDArray.empty(Shape(1, 3, resizedImg.height, resizedImg.width), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    sample
  }

  def preprocessStyleImage(path: String, shape: Shape, ctx: Context): NDArray = {
    val img = Image(new File(path))
    val resizedImg = img.scaleTo(shape(3), shape(2))
    val sample = NDArray.empty(Shape(1, 3, shape(2), shape(3)), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    sample
  }

  def clip(array: Array[Float]): Array[Float] = array.map { a =>
    if (a < 0) 0f
    else if (a > 255) 255f
    else a
  }

  def postprocessImage(img: NDArray): Image = {
    val datas = img.toArray
    val spatialSize = img.shape(2) * img.shape(3)
    val r = clip(datas.take(spatialSize).map(_ + 123.68f))
    val g = clip(datas.drop(spatialSize).take(spatialSize).map(_ + 116.779f))
    val b = clip(datas.takeRight(spatialSize).map(_ + 103.939f))
    val pixels = for (i <- 0 until spatialSize)
      yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
    Image(img.shape(3), img.shape(2), pixels.toArray)
  }

  def saveImage(img: NDArray, filename: String, radius: Int): Unit = {
    val out = postprocessImage(img)
    val gauss = GaussianBlurFilter(radius).op
    val result = Image(out.width, out.height)
    gauss.filter(out.awt, result.awt)
    result.output(filename)(JpegWriter())
  }
}
