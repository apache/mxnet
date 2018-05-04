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

package org.apache.mxnet.infer

import org.apache.mxnet.{Context, DataDesc, NDArray, Shape}

import scala.collection.mutable.ListBuffer

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on
import java.io.File

import javax.imageio.ImageIO


/**
  * A class for image classification tasks.
  * Contains helper methods.
  *
  * @param modelPathPrefix    Path prefix from where to load the model artifacts.
  *                           These include the symbol, parameters, and synset.txt.
  *                           Example: file://model-dir/resnet-152 (containing
  *                           resnet-152-symbol.json, resnet-152-0000.params, and synset.txt).
  * @param inputDescriptors   Descriptors defining the input node names, shape,
  *                           layout and type parameters
  * @param contexts           Device contexts on which you want to run inference; defaults to CPU
  * @param epoch              Model epoch to load; defaults to 0
  */
class ImageClassifier(modelPathPrefix: String,
                      inputDescriptors: IndexedSeq[DataDesc],
                      contexts: Array[Context] = Context.cpu(),
                      epoch: Option[Int] = Some(0))
                      extends Classifier(modelPathPrefix,
                      inputDescriptors, contexts, epoch) {

  protected[infer] val inputLayout = inputDescriptors.head.layout

  require(inputDescriptors.nonEmpty, "Please provide input descriptor")
  require(inputDescriptors.head.layout == "NCHW", "Provided layout doesn't match NCHW format")

  protected[infer] val inputShape = inputDescriptors.head.shape

  // Considering 'NCHW' as default layout when not provided
  // Else get axis according to the layout
  // [TODO] if layout is different than the bufferedImage layout,
  // transpose to match the inputdescriptor shape
  protected[infer] val batch = inputShape(inputLayout.indexOf('N'))
  protected[infer] val channel = inputShape(inputLayout.indexOf('C'))
  protected[infer] val height = inputShape(inputLayout.indexOf('H'))
  protected[infer] val width = inputShape(inputLayout.indexOf('W'))

  /**
    * To classify the image according to the provided model
    *
    * @param inputImage       Path prefix of the input image
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of (Label, Probability)
    */
  def classifyImage(inputImage: BufferedImage,
                    topK: Option[Int] = None): IndexedSeq[IndexedSeq[(String, Float)]] = {

    val scaledImage = ImageClassifier.reshapeImage(inputImage, width, height)
    val pixelsNDArray = ImageClassifier.bufferedImageToPixels(scaledImage, inputShape)
    inputImage.flush()
    scaledImage.flush()

    val output = super.classifyWithNDArray(IndexedSeq(pixelsNDArray), topK)

    handler.execute(pixelsNDArray.dispose())

    IndexedSeq(output(0))
  }

  /**
    * To classify batch of input images according to the provided model
    *
    * @param inputBatch       Input array of buffered images
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of (Label, Probability)
    */
  def classifyImageBatch(inputBatch: Traversable[BufferedImage], topK: Option[Int] = None):
  IndexedSeq[IndexedSeq[(String, Float)]] = {

    val imageBatch = ListBuffer[NDArray]()
    for (image <- inputBatch) {
      val scaledImage = ImageClassifier.reshapeImage(image, width, height)
      val pixelsNDArray = ImageClassifier.bufferedImageToPixels(scaledImage, inputShape)
      imageBatch += pixelsNDArray
    }
    val op = NDArray.concatenate(imageBatch)

    val result = super.classifyWithNDArray(IndexedSeq(op), topK)
    handler.execute(op.dispose())
    handler.execute(imageBatch.foreach(_.dispose()))

    result
  }

  private[infer] def getClassifier(modelPathPrefix: String,
                                     inputDescriptors: IndexedSeq[DataDesc],
                    contexts: Array[Context] = Context.cpu(),
                    epoch: Option[Int] = Some(0)): Classifier = {
    new Classifier(modelPathPrefix, inputDescriptors, contexts, epoch)
  }
}

object ImageClassifier {

  /**
    * Reshape the input image to a new shape
    *
    * @param img              Input image
    * @param newWidth         New width for rescaling
    * @param newHeight        New height for rescaling
    * @return                 Rescaled BufferedImage
    */
  def reshapeImage(img: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    val resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB)
    val g = resizedImage.createGraphics()
    g.drawImage(img, 0, 0, newWidth, newHeight, null)
    g.dispose()

    resizedImage
  }

  /**
    * Convert input BufferedImage to NDArray of input shape
    *
    * <p>
    * Note: Caller is responsible to dispose the NDArray
    * returned by this method after the use.
    * </p>
    * @param resizedImage     BufferedImage to get pixels from
    * @param inputImageShape  Input shape; for example for resnet it is (1,3,224,224).
                              Should be same as inputDescriptor shape.
    * @return                 NDArray pixels array
    */
  def bufferedImageToPixels(resizedImage: BufferedImage, inputImageShape: Shape): NDArray = {
    // Get height and width of the image
    val w = resizedImage.getWidth()
    val h = resizedImage.getHeight()

    // get an array of integer pixels in the default RGB color mode
    val pixels = resizedImage.getRGB(0, 0, w, h, null, 0, w)

    // 3 times height and width for R,G,B channels
    val result = new Array[Float](3 * h * w)

    var row = 0
    // copy pixels to array vertically
    while (row < h) {
      var col = 0
      // copy pixels to array horizontally
      while (col < w) {
        val rgb = pixels(row * w + col)
        // getting red color
        result(0 * h * w + row * w + col) = (rgb >> 16) & 0xFF
        // getting green color
        result(1 * h * w + row * w + col) = (rgb >> 8) & 0xFF
        // getting blue color
        result(2 * h * w + row * w + col) = rgb & 0xFF
        col += 1
      }
      row += 1
    }
    resizedImage.flush()

    // creating NDArray according to the input shape
    val pixelsArray = NDArray.array(result, shape = inputImageShape)
    pixelsArray
  }

  /**
    * Loads an input images from file
    * @param inputImagePath   Path of single input image
    * @return                 BufferedImage Buffered image
    */
  def loadImageFromFile(inputImagePath: String): BufferedImage = {
    val img = ImageIO.read(new File(inputImagePath))
    img
  }

  /**
    * Loads a batch of images from a folder
    * @param inputImageDirPath  Path to a folder of images
    * @return                   List of buffered images
    */
  def loadInputBatch(inputImagePaths: List[String]): Traversable[BufferedImage] = {
    inputImagePaths.map(path => ImageIO.read(new File(path)))
  }
}