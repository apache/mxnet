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

package ml.dmlc.mxnet.spark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class MXNetGeneralSuite extends SharedSparkContext {

  private def parseRawData(sc: SparkContext, path: String): RDD[LabeledPoint] = {
    val raw = sc.textFile(path)
    raw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }
  }

  test("run spark with MLP") {
    val trainData = parseRawData(sc,
      "/Users/nanzhu/code/mxnet/scala-package/spark/train.txt")
    val model = buildMlp().fit(trainData)
    assert(model != null)
  }

  test("run spark with LeNet") {
    val trainData = parseRawData(sc,
      "/Users/nanzhu/code/mxnet/scala-package/spark/train.txt")
    val model = buildLeNet().fit(trainData)
    assert(model != null)
  }
}
