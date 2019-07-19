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

package org.apache.mxnet

import java.io.File
import java.net.URL

import javax.imageio.ImageIO
import org.apache.commons.io.FileUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

class ImageSuite extends FunSuite with BeforeAndAfterAll {
  private var imLocation = ""
  private val logger = LoggerFactory.getLogger(classOf[ImageSuite])

  private def downloadUrl(url: String, filePath: String, maxRetry: Option[Int] = None) : Unit = {
    val tmpFile = new File(filePath)
    var retry = maxRetry.getOrElse(3)
    var success = false
    if (!tmpFile.exists()) {
      while (retry > 0 && !success) {
        try {
          FileUtils.copyURLToFile(new URL(url), tmpFile)
          success = true
        } catch {
          case e: Exception => retry -= 1
        }
      }
    } else {
      success = true
    }
    if (!success) throw new Exception(s"$url Download failed!")
  }

  override def beforeAll(): Unit = {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    imLocation = tempDirPath + "/inputImages/Pug-Cookie.jpg"
    downloadUrl("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg",
      imLocation)
  }

  test("Test load image") {
    val nd = Image.imRead(imLocation)
    logger.debug(s"OpenCV load image with shape: ${nd.shape}")
    require(nd.shape == Shape(576, 1024, 3), "image shape not Match!")
  }

  test("Test load image from Socket") {
    val url = new URL("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg")
    val inputStream = url.openStream
    val nd = Image.imDecode(inputStream)
    logger.debug(s"OpenCV load image with shape: ${nd.shape}")
    require(nd.shape == Shape(576, 1024, 3), "image shape not Match!")
  }

  test("Test resize image") {
    val nd = Image.imRead(imLocation)
    val resizeIm = Image.imResize(nd, 224, 224)
    logger.debug(s"OpenCV resize image with shape: ${resizeIm.shape}")
    require(resizeIm.shape == Shape(224, 224, 3), "image shape not Match!")
  }

  test("Test crop image") {
    val nd = Image.imRead(imLocation)
    val nd2 = Image.fixedCrop(nd, 0, 0, 224, 224)
    require(nd2.shape == Shape(224, 224, 3), "image shape not Match!")
  }

  test("Test apply border") {
    val nd = Image.imRead(imLocation)
    val nd2 = Image.copyMakeBorder(nd, 1, 1, 1, 1)
    require(nd2.shape == Shape(578, 1026, 3), s"image shape not Match!")
  }

  test("Test convert to Image") {
    val nd = Image.imRead(imLocation)
    val resizeIm = Image.imResize(nd, 224, 224)
    val tempDirPath = System.getProperty("java.io.tmpdir")
    val img = Image.toImage(resizeIm)
    ImageIO.write(img, "png", new File(tempDirPath + "/inputImages/out.png"))
    logger.debug(s"converted image stored in ${tempDirPath + "/inputImages/out.png"}")
  }

  test("Test draw Bounding box") {
    val buf = ImageIO.read(new File(imLocation))
    val box = Array(
      Map("xmin" -> 190, "xmax" -> 850, "ymin" -> 50, "ymax" -> 450),
      Map("xmin" -> 200, "xmax" -> 350, "ymin" -> 440, "ymax" -> 530)
    )
    val names = Array("pug", "cookie")
    Image.drawBoundingBox(buf, box, Some(names), fontSizeMult = Some(1.4f))
    val tempDirPath = System.getProperty("java.io.tmpdir")
    ImageIO.write(buf, "png", new File(tempDirPath + "/inputImages/out2.png"))
    logger.debug(s"converted image stored in ${tempDirPath + "/inputImages/out2.png"}")
    for (coord <- box) {
      val topLeft = buf.getRGB(coord("xmin"), coord("ymin"))
      val downLeft = buf.getRGB(coord("xmin"), coord("ymax"))
      val topRight = buf.getRGB(coord("xmax"), coord("ymin"))
      val downRight = buf.getRGB(coord("xmax"), coord("ymax"))
      require(downLeft == downRight)
      require(topRight == downRight)
    }
  }

}
