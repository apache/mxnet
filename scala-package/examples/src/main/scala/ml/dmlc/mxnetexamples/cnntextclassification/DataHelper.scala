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

package ml.dmlc.mxnetexamples.cnntextclassification

import scala.io.Source
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.DataInputStream
import java.io.InputStream
import ml.dmlc.mxnet.Random
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Shape

/**
 * @author Depeng Liang
 */
object DataHelper {

  def cleanStr(str: String): String = {
    str.replaceAll("[^A-Za-z0-9(),!?'`]", " ")
        .replaceAll("'s", " 's")
        .replaceAll("'ve", " 've")
        .replaceAll("n't", " n't")
        .replaceAll("'re", " 're")
        .replaceAll("'d", " 'd")
        .replaceAll("'ll", " 'll")
        .replaceAll(",", " , ")
        .replaceAll("!", " ! ")
        .replaceAll("\\(", " \\( ")
        .replaceAll("\\)", " \\) ")
        .replaceAll("\\?", " \\? ")
        .replaceAll(" {2,}", " ")
        .trim()
  }

  // Loads MR polarity data from files, splits the data into words and generates labels.
  // Returns split sentences and labels.
  def loadMRDataAndLabels(dataPath: String): (Array[Array[String]], Array[Float]) = {
    // load data from file
    val positiveExamples = {
      val lines = Source.fromFile(s"$dataPath/rt-polarity.pos").mkString.split("\n")
      lines.map(_.trim())
    }
    val negativeExamples = {
      val lines = Source.fromFile(s"$dataPath/rt-polarity.neg").mkString.split("\n")
      lines.map(_.trim())
    }
    // split by words
    val xText = {
      val tmp = positiveExamples ++ negativeExamples
      tmp.map(cleanStr(_)).map(_.split(" "))
    }
    // generate labels
    val positiveLabels = (1 to positiveExamples.length).map(x => 1).toArray
    val negativeLabels = (1 to negativeExamples.length).map(x => 0).toArray
    val y = positiveLabels ++ negativeLabels
    (xText, y.map(_.toFloat))
  }

  // Pads all sentences to the same length. The length is defined by the longest sentence.
  // Returns padded sentences.
  def padSentences(sentences: Array[Array[String]],
    paddingWord: String = "</s>"): Array[Array[String]] = {
    val sequenceLength = (-1 /: sentences.map(_.length)){ (max, len) =>
      if (max < len) len else max
    }
    val paddedSetences = sentences.map { sentence =>
      val numPadding = sequenceLength - sentence.length
      sentence ++ (1 to numPadding).map(x => paddingWord)
    }
    paddedSetences
  }

  def loadPretrainedWord2vec(inFile: String): (Int, Map[String, Array[Float]]) = {
    val lines = Source.fromFile(inFile).mkString.mkString.split("\n")
    val (vocabSize, dim) = {
      val head = lines(0).split(" ").map(_.toInt)
      (head(0), head(1))
    }
    val word2vec = lines.drop(1).map { line =>
      val tks = line.trim().split(" ")
      tks(0) -> tks.drop(1).map(_.toFloat)
    }.toMap
    (dim, word2vec)
  }

  def readString(dis: DataInputStream): String = {
    val MAX_SIZE = 50
    var bytes = new Array[Byte](MAX_SIZE)
    var b = dis.readByte()
    var i = -1
    val sb = new StringBuilder()
    while (b != 32 && b != 10) {
      i = i + 1
      bytes(i) = b
      b = dis.readByte()
      if (i == 49) {
        sb.append(new String(bytes))
        i = -1
        bytes = new Array[Byte](MAX_SIZE)
      }
    }
    sb.append(new String(bytes, 0, i + 1))
    sb.toString()
  }

  def getFloat(b: Array[Byte]): Float = {
    var accum = 0
    accum = accum | (b(0) & 0xff) << 0
    accum = accum | (b(1) & 0xff) << 8
    accum = accum | (b(2) & 0xff) << 16
    accum = accum | (b(3) & 0xff) << 24
    java.lang.Float.intBitsToFloat(accum).toFloat
  }

  def readFloat(is: InputStream): Float = {
    val bytes = new Array[Byte](4)
    is.read(bytes)
    getFloat(bytes)
  }

  // Reference https://github.com/NLPchina/Word2VEC_java
  def loadGoogleModel(path: String): (Int, Map[String, Array[Float]]) = {
    val bis = new BufferedInputStream(new FileInputStream(path))
    val dis = new DataInputStream(bis)
    val wordSize = Integer.parseInt(readString(dis))
    val dim = Integer.parseInt(readString(dis))
    var word2vec = Map[String, Array[Float]]()
    for (i <- 0 until wordSize) {
      val word = readString(dis)
      val vectors = (1 to dim).map(j => readFloat(dis)).toArray
      word2vec += word -> vectors
    }
    bis.close()
    dis.close()
    (dim, word2vec)
  }

  // Map sentences and labels to vectors based on a pretrained word2vec.
  def buildInputDataWithWord2vec(sentences: Array[Array[String]], embeddingSize: Int,
    word2vec: Map[String, Array[Float]]): Array[Array[Array[Float]]] = {
    val xVec = sentences.map { sentence =>
      sentence.map { word =>
        if (word2vec.contains(word)) word2vec(word)
        else Random.uniform(-0.25f, 0.25f, Shape(embeddingSize), Context.cpu()).toArray
      }
    }
    xVec
  }

  def loadMSDataWithWord2vec(dataPath: String, embeddingSize: Int,
    word2vec: Map[String, Array[Float]]): (Array[Array[Array[Float]]], Array[Float]) = {
    // loads the MR dataset
    val (sentences, labels) = loadMRDataAndLabels(dataPath)
    val sentencesPadded = padSentences(sentences)
    (buildInputDataWithWord2vec(sentencesPadded, embeddingSize, word2vec), labels)
  }
}
