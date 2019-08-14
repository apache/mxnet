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
package sample
import org.apache.mxnet._

object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("hello World")
    val arr = NDArray.ones(2, 3)
    println(arr.shape)
    println("test")
    for (rootScale <- 1 to 3) {
      for (scale <- 1 to 3) {
        for (numShape <- 1 to 3) {
          for (base <- 1 to 3) {
            val shapes = (0 until numShape).map(i =>
              Shape(1, 3, base * rootScale * Math.pow(scale, numShape - 1 - i).toInt,
                base * rootScale * Math.pow(scale, numShape - 1 - i).toInt))
            checkNearestUpSamplingWithShape(shapes, scale, rootScale)
          }
        }
      }
    }
    println("upsampling changes")
  }

  def checkNearestUpSamplingWithShape(shapes: Seq[Shape],
                                              scale: Int,
                                              rootScale: Int): Unit = {
    val arr = shapes.zipWithIndex.map { case (shape, i) =>
      (s"arg_$i", Random.uniform(-10, 10, shape))
    }.toMap

    val arrGrad = shapes.zipWithIndex.map { case (shape, i) =>
      (s"arg_$i", NDArray.zeros(shape))
    }.toMap

    val upArgs = (0 until shapes.size).map(i => Symbol.Variable(s"arg_$i"))
    val up = Symbol.UpSampling()(upArgs: _*)(Map("sample_type" -> "nearest", "scale" -> rootScale))
    val exe = up.bind(Context.cpu(), args = arr, argsGrad = arrGrad)
    exe.forward(isTrain = true)
    exe.backward(exe.outputs)
    for (k <- 0 until shapes.size) {
      val name = s"arg_$k"
      val expected =
        arr(name).toArray.map(_ * Math.pow(rootScale, 2).toFloat * Math.pow(scale, 2 * k).toFloat)
      val real = arrGrad(name).toArray
      (expected zip real) foreach { case (e, r) =>
        assert(r === e +- 0.1f)
      }
    }
  }

  test("nearest upsampling") {
    for (rootScale <- 1 to 3) {
      for (scale <- 1 to 3) {
        for (numShape <- 1 to 3) {
          for (base <- 1 to 3) {
            val shapes = (0 until numShape).map(i =>
              Shape(1, 3, base * rootScale * Math.pow(scale, numShape - 1 - i).toInt,
                base * rootScale * Math.pow(scale, numShape - 1 - i).toInt))
            checkNearestUpSamplingWithShape(shapes, scale, rootScale)
          }
        }
      }
    }
  }
}