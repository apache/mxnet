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

package ml.dmlc.mxnet.infer

import ml.dmlc.mxnet._

import scala.collection.mutable.ListBuffer

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on
import java.io.File
import javax.imageio.ImageIO


/**
  * A class for classifier tasks
  *
  * @param modelPathPrefix  PathPrefix from where to load the symbol, parameters and synset.txt
  *                         Example: file://model-dir/resnet-152(containing resnet-152-symbol.json
  *                         file://model-dir/synset.txt
  * @param inputDescriptors Descriptors defining the input node names, shape,
  *                         layout and Type parameters
  */
class ImageClassifier(modelPathPrefix: String,
                      inputDescriptors: IndexedSeq[DataDesc])
                      extends Classifier(modelPathPrefix,
                      inputDescriptors) {

  val classifier: Classifier = getClassifier(modelPathPrefix, inputDescriptors)

  // Loading image from file
  def loadImageFromFile(inputImagePath: String): BufferedImage = {
      val img = ImageIO.read(new File(inputImagePath))
      img
  }

  /**
    * Loading input batch of images
    * @param inputImageDirPath
    * @return List of buffered images
    */
  def loadInputBatch(inputImageDirPath: String): List[BufferedImage] = {
    val dir = new File(inputImageDirPath)
    val inputBatch = ListBuffer[BufferedImage]()
    for (imgFile: File <- dir.listFiles()){
      val img = ImageIO.read(imgFile)
      inputBatch += img
    }
    inputBatch.toList
  }

  /**
    * Reshape the input image to new shape
    *
    * @param img       image
    * @param newWidth  rescale to new width
    * @param newHeight rescale to new height
    * @return Rescaled BufferedImage
    */
  def getScaledImage(img: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    val resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB)
    val g = resizedImage.createGraphics()
    g.drawImage(img, 0, 0, newWidth, newHeight, null)
    g.dispose()

    resizedImage
  }

  /**
    * Read image file from provided path
    *
    * @param resizedImage BufferedImage to get pixels from
    * @return NDArray pixels array
    */
  def getPixelsFromImage(resizedImage: BufferedImage): NDArray = {
    val w = resizedImage.getWidth
    val h = resizedImage.getHeight

    val pixels = new ListBuffer[Float]()

    for (x <- 0 until h) {
      for (y <- 0 until w) {
        val color = resizedImage.getRGB(y, x)
        val red = (color & 0xff0000) >> 16
        val green = (color & 0xff00) >> 8
        val blue = color & 0xff
        pixels += red
        pixels += green
        pixels += blue
      }
    }

    val reshaped_pixels = NDArray.array(pixels.toArray, shape = Shape(224, 224, 3))

    val swapped_axis = NDArray.swapaxes(reshaped_pixels, 0, 2)
    val pixelsNdarray = NDArray.swapaxes(swapped_axis, 1, 2)

    pixelsNdarray
  }

  /**
    * To classify the image according to the provided model
    *
    * @param inputImage PathPrefix of the input image
    * @param topK Get top k elements with maximum probability
    * @return List of list of tuples of (class, probability)
    */
  def classifyImage(inputImage: BufferedImage,
                        topK: Option[Int] = None): IndexedSeq[List[(String, Float)]] = {

    val width = inputDescriptors(0).shape(2)
    val height = inputDescriptors(0).shape(3)

    val scaledImage = this.getScaledImage(inputImage, width, height)
    val pixelsNdarray = this.getPixelsFromImage(scaledImage)

    val input = IndexedSeq(pixelsNdarray.reshape(inputDescriptors(0).shape))

    val output = super.classifyWithNDArray(input, topK)

    IndexedSeq(output(0))
  }

  /**
    * To classify batch of input images according to the provided model
    * @param inputBatch Input batch of Buffered images
    * @param topK Get top k elements with maximum probability
    * @return List of list of tuples of (class, probability)
    */
  // [TODO] change to batched ndarrays to improve performance
  def classifyImageBatch(inputBatch: Traversable[BufferedImage], topK: Option[Int] = None):
  List[List[(String, Float)]] = {
    val result = ListBuffer[List[(String, Float)]]()
    for (image <- inputBatch) {
      result += this.classifyImage(image, topK)(0)
    }
    result.toList
  }

  def getClassifier(modelPathPrefix: String, inputDescriptors: IndexedSeq[DataDesc]): Classifier = {
    new Classifier(modelPathPrefix, inputDescriptors)
  }

}
