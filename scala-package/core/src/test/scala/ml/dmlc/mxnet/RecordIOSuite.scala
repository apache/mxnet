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

package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import java.io._

class RecordIOSuite extends FunSuite with BeforeAndAfterAll {
  test("test RecordIO") {
    val fRec = File.createTempFile("tmpFile", ".tmp")
    val N = 255

    val writer = new MXRecordIO(fRec.getAbsolutePath, MXRecordIO.IOWrite)
    for (i <- 0 until N) {
      writer.write(s"$i")
    }
    writer.close()

    val reader = new MXRecordIO(fRec.getAbsolutePath, MXRecordIO.IORead)
    for (i <- 0 until N) {
      val res = reader.read()
      assert(res === s"$i")
    }
  }

  test("test IndexedRecordIO") {
    val fIdxRec = File.createTempFile("tmpIdxFile", ".tmp")
    val fIdx = File.createTempFile("tmpIdx", ".tmp")
    val N = 255

    val writer = new MXIndexedRecordIO(fIdx.getAbsolutePath,
        fIdxRec.getAbsolutePath, MXRecordIO.IOWrite)
    for (i <- 0 until N) {
      writer.writeIdx(i, s"$i")
    }
    writer.close()

    val reader = new MXIndexedRecordIO(fIdx.getAbsolutePath,
        fIdxRec.getAbsolutePath, MXRecordIO.IORead)
    var keys = reader.keys().map(_.asInstanceOf[Int]).toList.sorted
    assert(keys.zip(0 until N).forall(x => x._1 == x._2))
    keys = scala.util.Random.shuffle(keys)
    for (k <- keys) {
      val res = reader.readIdx(k)
      assert(res === s"$k")
    }
  }

  test("test RecordIOPackLabel") {
    val fRec = File.createTempFile("tmpFile", ".tmp")
    val N = 255

    val charsDigits =
      (0 until 26).map(x => ('A' + x).toChar.toString ).toArray ++ (0 to 9).map(_.toString)

    for (i <- 1 until N) {
      for (j <- 0 until N) {
        val content = {
          val idx = scala.util.Random.shuffle(charsDigits.indices.toList).take(j)
          idx.map(charsDigits(_)).mkString
        }
        val label = (0 until i).map(x => scala.util.Random.nextFloat()).toArray
        val header = MXRecordIO.IRHeader(0, label, 0, 0)
        val s = MXRecordIO.pack(header, content)
        val (rHeader, rContent) = MXRecordIO.unpack(s)
        assert(label === rHeader.label)
        assert(content === rContent)
      }
    }
  }
}
