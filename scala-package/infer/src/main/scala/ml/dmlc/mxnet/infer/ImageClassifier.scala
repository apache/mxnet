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

import ml.dmlc.mxnet.{DataDesc, NDArray, Shape}

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

  val inputLayout = inputDescriptors(0).layout

  val inputShape = inputDescriptors(0).shape

  // Considering 'NCHW' as default layout when not provided
  // Else get axis according to the layout
  val batch = inputShape(if (inputLayout.indexOf('N')<0) 0 else inputLayout.indexOf('N'))
  val channel = inputShape(if (inputLayout.indexOf('C')<0) 1 else inputLayout.indexOf('C'))
  val height = inputShape(if (inputLayout.indexOf('H')<0) 2 else inputLayout.indexOf('H'))
  val width = inputShape(if (inputLayout.indexOf('W')<0) 3 else inputLayout.indexOf('W'))

  /**
    * To classify the image according to the provided model
    *
    * @param inputImage PathPrefix of the input image
    * @param topK Get top k elements with maximum probability
    * @return List of list of tuples of (class, probability)
    */
  def classifyImage(inputImage: BufferedImage,
                    topK: Option[Int] = None): IndexedSeq[IndexedSeq[(String, Float)]] = {

    val scaledImage = ImageClassifier.reshapeImage(inputImage, width, height)
    val pixelsNdarray = ImageClassifier.bufferedImageToPixels(scaledImage, inputShape)

    val output = super.classifyWithNDArray(IndexedSeq(pixelsNdarray), topK)

    handler.execute(pixelsNdarray.dispose())

    IndexedSeq(output(0))
  }

  /**
    * To classify batch of input images according to the provided model
    * @param inputBatch Input batch of Buffered images
    * @param topK Get top k elements with maximum probability
    * @return List of list of tuples of (class, probability)
    */
  def classifyImageBatch(inputBatch: Traversable[BufferedImage], topK: Option[Int] = None):
  IndexedSeq[IndexedSeq[(String, Float)]] = {

    val imageBatch = ListBuffer[NDArray]()
    for (image <- inputBatch) {
      val scaledImage = ImageClassifier.reshapeImage(image, width, height)
      val pixelsNdarray = ImageClassifier.bufferedImageToPixels(scaledImage, inputShape)
      imageBatch += pixelsNdarray
    }
    val op = NDArray.concatenate(imageBatch)

    val result = super.classifyWithNDArray(IndexedSeq(op), topK)

    result
  }

  def getClassifier(modelPathPrefix: String, inputDescriptors: IndexedSeq[DataDesc]): Classifier = {
    new Classifier(modelPathPrefix, inputDescriptors)
  }
}

object ImageClassifier {

  /**
    * Reshape the input image to new shape
    *
    * @param img       input image
    * @param newWidth  rescale to new width
    * @param newHeight rescale to new height
    * @return Rescaled BufferedImage
    */
  def reshapeImage(img: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
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
    * @param inputImageShape Should be same as inputDescriptor shape
    * @return NDArray pixels array
    */
  def bufferedImageToPixels(resizedImage: BufferedImage, inputImageShape: Shape): NDArray = {
    val w = resizedImage.getWidth()
    val h = resizedImage.getHeight()
    val pixels = resizedImage.getRGB(0, 0, w, h, null, 0, w)
    val result = new Array[Float](3 * h * w)

    var row = 0
    while (row < h) {
      var col = 0
      while (col < w) {
        val rgb = pixels(row * w + col)
        result(0 * h * w + row * w + col) = (rgb >> 16) & 0xFF
        result(1 * h * w + row * w + col) = (rgb >> 8) & 0xFF
        result(2 * h * w + row * w + col) = rgb & 0xFF
        col += 1
      }
      row += 1
    }

    val pixelsArray = NDArray.array(result, shape = inputImageShape)
    pixelsArray
  }

  /**
    * Loading input batch of images
    * @param inputImagePath Path of single input image
    * @return BufferedImage Buffered image
    */
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
}
