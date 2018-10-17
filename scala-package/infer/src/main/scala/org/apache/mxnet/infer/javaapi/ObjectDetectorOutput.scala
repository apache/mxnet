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

class ObjectDetectorOutput (className: String, args: Array[Float]){

  private val probability = args(0)
  private val xMin = args(1)
  private val xMax = args(2)
  private val yMin = args(3)
  private val yMax = args(4)

  def getClassName: String = className

  def getProbability: Float = probability

  def getXMin: Float= xMin

  def getXMax: Float = xMax

  def getYMin: Float = yMin

  def getYMax: Float = yMax

}
