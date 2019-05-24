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

package org.apache.mxnet.infer.javaapi

/**
  * The ObjectDetectorOutput class is a simple POJO helper class that is used to simplify
  * the interactions with ObjectDetector predict results. The class stores the bounding box
  * coordinates, name of preicted class, and the probability.
  */


class ObjectDetectorOutput (className: String, args: Array[Float]){

  /**
    * Gets the predicted class's name.
    *
    * @return       String representing the name of the predicted class
    */
  def getClassName: String = className

  /**
    * Gets the probability of the predicted class.
    *
    * @return       Float representing the probability of predicted class
    */
  def getProbability: Float = args(0)

  /**
    * Gets the minimum X coordinate for the bounding box containing the predicted object.
    *
    * @return       Float of the min X coordinate for the object bounding box
    */
  def getXMin: Float = args(1)

  /**
    * Gets the maximum X coordinate for the bounding box containing the predicted object.
    *
    * @return       Float of the max X coordinate for the object bounding box
    */
  def getXMax: Float = args(3)

  /**
    * Gets the minimum Y coordinate for the bounding box containing the predicted object.
    *
    * @return       Float of the min Y coordinate for the object bounding box
    */
  def getYMin: Float = args(2)

  /**
    * Gets the maximum Y coordinate for the bounding box containing the predicted object.
    *
    * @return       Float of the max Y coordinate for the object bounding box
    */
  def getYMax: Float = args(4)

}
