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

package ml.dmlc.mxnetexamples.profiler

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import java.io.File
import ml.dmlc.mxnet.Profiler
import ml.dmlc.mxnet.Random
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Context

/**
 * @author Depeng Liang
 */
object ProfilerNDArray {
  private val logger = LoggerFactory.getLogger(classOf[ProfilerNDArray])

  def testBroadcast(): Unit = {
    val sampleNum = 1000
    def testBroadcastTo(): Unit = {
      for (i <- 0 until sampleNum) {
        val nDim = scala.util.Random.nextInt(2) + 1
        val targetShape = Shape((0 until nDim).map(i => scala.util.Random.nextInt(10) + 1))
        val shape = targetShape.toArray.map { s =>
            if (scala.util.Random.nextInt(2) == 1) 1
            else s
        }
        val dat = NDArray.empty(shape: _*)
        val randomRet = (0 until shape.product)
          .map(r => scala.util.Random.nextFloat() - 0.5f).toArray
        dat.set(randomRet)
        val ndArrayRet = NDArray.broadcast_to(Map("shape" -> targetShape))(dat).get
        require(ndArrayRet.shape == targetShape)
        val err = {
          // implementation of broadcast
          val ret = {
            (randomRet /: shape.zipWithIndex.reverse){ (acc, elem) => elem match { case (s, i) =>
              if (s != targetShape(i)) {
                acc.grouped(shape.takeRight(shape.length - i).product).map {g =>
                  (0 until targetShape(i)).map(x => g).flatten
                }.flatten.toArray
              } else acc
            }}
          }
          val tmp = ndArrayRet.toArray.zip(ret).map{ case (l, r) => Math.pow(l - r, 2) }
          tmp.sum / tmp.length
        }
        require(err < 1E-8)
        ndArrayRet.dispose()
        dat.dispose()
      }
    }
    testBroadcastTo()
  }

  def randomNDArray(dim: Int): NDArray = {
    val tmp = Math.pow(1000, 1.0 / dim).toInt
    val shape = Shape((0 until dim).map(d => scala.util.Random.nextInt(tmp) + 1))
    Random.uniform(-10f, 10f, shape)
  }

  def testNDArraySaveload(): Unit = {
    val maxDim = 5
    val nRepeat = 10
    val fileName = s"${System.getProperty("java.io.tmpdir")}/tmpList.bin"
    for (repeat <- 0 until nRepeat) {
      try {
        val data = (0 until 10).map(i => randomNDArray(scala.util.Random.nextInt(4) + 1))
        NDArray.save(fileName, data)
        val data2 = NDArray.load2Array(fileName)
        require(data.length == data2.length)
        for ((x, y) <- data.zip(data2)) {
          val tmp = x - y
          require(tmp.toArray.sum == 0)
          tmp.dispose()
        }
        val dMap = data.zipWithIndex.map { case (arr, i) =>
          s"NDArray xx $i" -> arr
        }.toMap
        NDArray.save(fileName, dMap)
         val dMap2 = NDArray.load2Map(fileName)
         require(dMap.size == dMap2.size)
         for ((k, x) <- dMap) {
           val y = dMap2(k)
           val tmp = x - y
           require(tmp.toArray.sum == 0)
           tmp.dispose()
         }
        data.foreach(_.dispose())
      } finally {
        val file = new File(fileName)
        file.delete()
      }
    }
  }

  def testNDArrayCopy(): Unit = {
    val c = Random.uniform(-10f, 10f, Shape(10, 10))
    val d = c.copyTo(Context.cpu(0))
    val tmp = c - d
    require(tmp.toArray.map(Math.abs).sum == 0)
    c.dispose()
    d.dispose()
  }

  def reldiff(a: NDArray, b: NDArray): Float = {
    val diff = NDArray.sum(NDArray.abs(a - b)).toScalar
    val norm = NDArray.sum(NDArray.abs(a)).toScalar
    diff / norm
  }

  def reldiff(a: Array[Float], b: Array[Float]): Float = {
    val diff =
      (a zip b).map { case (aElem, bElem) => Math.abs(aElem - bElem) }.sum
    val norm: Float = a.reduce(Math.abs(_) + Math.abs(_))
    diff / norm
  }

  def testNDArrayNegate(): Unit = {
    val rand = Random.uniform(-10f, 10f, Shape(2, 3, 4))
    val npy = rand.toArray
    val arr = NDArray.empty(Shape(2, 3, 4))
    arr.set(npy)
    require(reldiff(npy, arr.toArray) < 1e-6f)
    val negativeArr = -arr
    require(reldiff(npy.map(_ * -1f), negativeArr.toArray) < 1e-6f)
    // a final check to make sure the negation (-) is not implemented
    // as inplace operation, so the contents of arr does not change after
    // we compute (-arr)
    require(reldiff(npy, arr.toArray) < 1e-6f)
    rand.dispose()
    arr.dispose()
    negativeArr.dispose()
  }

  def testNDArrayScalar(): Unit = {
    val c = NDArray.empty(10, 10)
    val d = NDArray.empty(10, 10)
    c.set(0.5f)
    d.set(1.0f)
    d -= c * 2f / 3f * 6f
    c += 0.5f
    require(c.toArray.sum - 100f < 1e-5f)
    require(d.toArray.sum + 100f < 1e-5f)
    c.set(2f)
    require(c.toArray.sum - 200f < 1e-5f)
    d.set(-c + 2f)
    require(d.toArray.sum < 1e-5f)
    c.dispose()
    d.dispose()
  }

  def testClip(): Unit = {
    val shape = Shape(10)
    val A = Random.uniform(-10f, 10f, shape)
    val B = NDArray.clip(A, -2f, 2f)
    val B1 = B.toArray
    require(B1.forall { x => x >= -2f && x <= 2f })
  }

  def testDot(): Unit = {
    val a = Random.uniform(-3f, 3f, Shape(3, 4))
    val b = Random.uniform(-3f, 3f, Shape(4, 5))
    val c = NDArray.dot(a, b)
    val A = a.toArray.grouped(4).toArray
    val B = b.toArray.grouped(5).toArray
    val C = (Array[Array[Float]]() /: A)((acc, row) => acc :+ row.zip(B).map(z =>
                z._2.map(_ * z._1)).reduceLeft(_.zip(_).map(x => x._1 + x._2))).flatten
    require(reldiff(c.toArray, C) < 1e-5f)
    a.dispose()
    b.dispose()
    c.dispose()
  }

  def testNDArrayOnehot(): Unit = {
    val shape = Shape(100, 20)
    var npy = (0 until shape.product).toArray.map(_.toFloat)
    val arr = NDArray.empty(shape)
    arr.set(npy)
    val nRepeat = 3
    for (repeat <- 0 until nRepeat) {
      val indices = (0 until shape(0)).map(i => scala.util.Random.nextInt(shape(1)))
      npy = npy.map(i => 0f)
      for (i <- 0 until indices.length) npy(i * shape(1) + indices(i)) = 1f
      val ind = NDArray.empty(shape(0))
      ind.set(indices.toArray.map(_.toFloat))
      NDArray.onehotEncode(ind, arr)
      require(arr.toArray.zip(npy).map(x => x._1 - x._2).sum == 0f)
      ind.dispose()
    }
    arr.dispose()
  }

  def main(args: Array[String]): Unit = {
    val eray = new ProfilerNDArray
    val parser: CmdLineParser = new CmdLineParser(eray)
    try {
      parser.parseArgument(args.toList.asJava)

      val path = s"${eray.outputPath}${File.separator}${eray.profilerName}"
      Profiler.profilerSetConfig(mode = eray.profilerMode, fileName = path)
      logger.info(s"profile file save to $path")

      Profiler.profilerSetState("run")
      testBroadcast()
      testNDArraySaveload()
      testNDArrayCopy()
      testNDArrayNegate()
      testNDArrayScalar()
      testClip()
      testDot()
      testNDArrayOnehot()
      Profiler.profilerSetState("stop")

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ProfilerNDArray {
  @Option(name = "--profiler-mode", usage = "the profiler mode, can be \"symbolic\" or \"all\".")
  private val profilerMode: String = "all"
  @Option(name = "--output-path", usage = "the profile file output directory.")
  private val outputPath: String = "."
  @Option(name = "--profile-filename", usage = "the profile file name.")
  private val profilerName: String = "profile_ndarray.json"
}
