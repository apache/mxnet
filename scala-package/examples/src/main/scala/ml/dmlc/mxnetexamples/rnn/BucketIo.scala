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


package ml.dmlc.mxnetexamples.rnn

import ml.dmlc.mxnet.{DataBatch, DataIter, NDArray, Shape}
import org.slf4j.LoggerFactory
import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random
import scala.collection.mutable

/**
 * @author Depeng Liang
 */
object BucketIo {

  type Text2Id = (String, Map[String, Int]) => Array[Int]
  type ReadContent = String => String

  def defaultReadContent(path: String): String = {
    Source.fromFile(path).mkString.replaceAll("\\. |\n", " <eos> ")
  }

  def defaultBuildVocab(path: String): Map[String, Int] = {
    val content = defaultReadContent(path).split(" ")
    var idx = 1 // 0 is left for zero - padding
    val vocab = mutable.Map.empty[String, Int]
    vocab.put(" ", 0) // put a dummy element here so that len (vocab) is correct
    content.foreach(word =>
      if (word.length > 0 && !vocab.contains(word)) {
        vocab.put(word, idx)
        idx += 1
      }
    )
    vocab.toMap
  }

  def defaultText2Id(sentence: String, theVocab: Map[String, Int]): Array[Int] = {
    val words = {
      val tmp = sentence.split(" ").filter(_.length() > 0)
      for (w <- tmp) yield theVocab(w)
    }
    words.toArray
  }

  def defaultGenBuckets(sentences: Array[String], batchSize: Int,
                        theVocab: Map[String, Int]): IndexedSeq[Int] = {
    val lenDict = scala.collection.mutable.Map[Int, Int]()
    var maxLen = -1
    for (sentence <- sentences) {
      val wordsLen = defaultText2Id(sentence, theVocab).length
      if (wordsLen > 0) {
        if (wordsLen > maxLen) {
          maxLen = wordsLen
        }
        if (lenDict.contains(wordsLen)) {
          lenDict(wordsLen) = lenDict(wordsLen) + 1
        } else {
          lenDict += wordsLen -> 1
        }
      }
    }

    var tl = 0
    val buckets = ArrayBuffer[Int]()
    lenDict.foreach {
      case (l, n) =>
        if (n + tl >= batchSize) {
          buckets.append(l)
          tl = 0
        } else tl += n
    }
    if (tl  > 0) buckets.append(maxLen)
    buckets
  }

  class BucketSentenceIter(
      path: String, vocab: Map[String, Int], var buckets: IndexedSeq[Int],
      _batchSize: Int, private val initStates: IndexedSeq[(String, (Int, Int))],
      seperateChar: String = " <eos> ", text2Id: Text2Id = defaultText2Id,
      readContent: ReadContent = defaultReadContent) extends DataIter {

    private val logger = LoggerFactory.getLogger(classOf[BucketSentenceIter])

    private val content = readContent(path)
    private val sentences = content.split(seperateChar)

    if (buckets.length == 0) {
      buckets = defaultGenBuckets(sentences, batchSize, vocab)
    }
    buckets = buckets.sorted
    // pre-allocate with the largest bucket for better memory sharing
    private val _defaultBucketKey = (buckets(0) /: buckets.drop(1)) { (max, elem) =>
      if (max < elem) elem else max
    }
    override def defaultBucketKey: AnyRef = _defaultBucketKey.asInstanceOf[AnyRef]
    // we just ignore the sentence it is longer than the maximum
    // bucket size here
    private val data = buckets.indices.map(x => Array[Array[Float]]()).toArray
    for (sentence <- sentences) {
      val ids = text2Id(sentence, vocab)
      if (ids.length > 0) {
        import scala.util.control.Breaks._
        breakable { buckets.indices.foreach { idx =>
          if (buckets(idx) >= ids.length) {
            data(idx) = data(idx) :+
            (ids.map(_.toFloat) ++ Array.fill[Float](buckets(idx) - ids.length)(0f))
            break()
          }
        }}
      }
    }

    // Get the size of each bucket, so that we could sample
    // uniformly from the bucket
    private val bucketSizes = data.map(_.length)
    logger.info("Summary of dataset ==================")
    buckets.zip(bucketSizes).foreach {
      case (bkt, size) => logger.info(s"bucket of len $bkt : $size samples")
    }

     // make a random data iteration plan
     // truncate each bucket into multiple of batch-size
    private var bucketNBatches = Array[Int]()
    for (i <- data.indices) {
      bucketNBatches = bucketNBatches :+ (data(i).length / _batchSize)
      data(i) = data(i).take(bucketNBatches(i) * _batchSize)
    }

    private val bucketPlan = {
      val plan = bucketNBatches.zipWithIndex.map(x => Array.fill[Int](x._1)(x._2)).flatten
      Random.shuffle(plan.toList).toArray
    }

    private val bucketIdxAll = data.map(_.length).map(l =>
      Random.shuffle((0 until l).toList).toArray)
    private val bucketCurrIdx = data.map(x => 0)

    private val dataBuffer = ArrayBuffer[NDArray]()
    private val labelBuffer = ArrayBuffer[NDArray]()
    for (iBucket <- data.indices) {
      dataBuffer.append(NDArray.zeros(_batchSize, buckets(iBucket)))
      labelBuffer.append(NDArray.zeros(_batchSize, buckets(iBucket)))
    }

    private val initStateArrays = initStates.map(x => NDArray.zeros(x._2._1, x._2._2))

    private val _provideData = { val tmp = ListMap("data" -> Shape(_batchSize, _defaultBucketKey))
      tmp ++ initStates.map(x => x._1 -> Shape(x._2._1, x._2._2))
    }
    private val _provideLabel = ListMap("softmax_label" -> Shape(_batchSize, _defaultBucketKey))

    private var iBucket = 0

    override def next(): DataBatch = {
      if (!hasNext) throw new NoSuchElementException
      val bucketIdx = bucketPlan(iBucket)
      val dataBuf = dataBuffer(bucketIdx)
      val iIdx = bucketCurrIdx(bucketIdx)
      val idx = bucketIdxAll(bucketIdx).slice(iIdx, iIdx + _batchSize)
      bucketCurrIdx(bucketIdx) = bucketCurrIdx(bucketIdx) + _batchSize

      val datas = idx.map(i => data(bucketIdx)(i))
      for (sentence <- datas) {
        require(sentence.length == buckets(bucketIdx))
      }
      dataBuf.set(datas.flatten)

      val labelBuf = labelBuffer(bucketIdx)
      val labels = idx.map(i => data(bucketIdx)(i).drop(1) :+ 0f)
      labelBuf.set(labels.flatten)

      iBucket += 1
      val batchProvideData = { val tmp = ListMap("data" -> dataBuf.shape)
        tmp ++ initStates.map(x => x._1 -> Shape(x._2._1, x._2._2))
      }
      val batchProvideLabel = ListMap("softmax_label" -> labelBuf.shape)
      new DataBatch(IndexedSeq(dataBuf) ++ initStateArrays,
                    IndexedSeq(labelBuf),
                    getIndex(),
                    getPad(),
                    this.buckets(bucketIdx).asInstanceOf[AnyRef],
                    batchProvideData, batchProvideLabel)
    }

    /**
     * reset the iterator
     */
    override def reset(): Unit = {
      iBucket = 0
      bucketCurrIdx.indices.foreach(i => bucketCurrIdx(i) = 0)
    }

    override def batchSize: Int = _batchSize

    /**
     * get data of current batch
     * @return the data of current batch
     */
    override def getData(): IndexedSeq[NDArray] = IndexedSeq(dataBuffer(bucketPlan(iBucket)))

    /**
     * Get label of current batch
     * @return the label of current batch
     */
    override def getLabel(): IndexedSeq[NDArray] = IndexedSeq(labelBuffer(bucketPlan(iBucket)))

    /**
     * the index of current batch
     * @return
     */
    override def getIndex(): IndexedSeq[Long] = IndexedSeq[Long]()

    // The name and shape of label provided by this iterator
    override def provideLabel: ListMap[String, Shape] = this._provideLabel

    /**
     * get the number of padding examples
     * in current batch
     * @return number of padding examples in current batch
     */
    override def getPad(): Int = 0

    // The name and shape of data provided by this iterator
    override def provideData: ListMap[String, Shape] = this._provideData

    override def hasNext: Boolean = {
      iBucket < bucketPlan.length
    }
  }
}
