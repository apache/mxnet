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

package org.apache.mxnet.util

import org.apache.mxnet.{NDArray, Shape}

/**
  * A visualize helper class to see the internal structure
  * of mxnet data-structure
  */
object Visualize {

  /**
    * Visualize the internal structure of NDArray
    * @return String that show the structure
    */
  def toString(nd : NDArray): String = {
    buildStringHelper(nd, nd.shape.length) + "\n"
  }
  /**
    * Helper function to create formatted NDArray output
    * The NDArray will be represented in a reduced version if too large
    * @param nd NDArray as the input
    * @param totalSpace totalSpace of the lowest dimension
    * @return String format of NDArray
    */
  private def buildStringHelper(nd : NDArray, totalSpace : Int) : String = {
    var result = ""
    val THRESHOLD = 100000      // longest NDArray to show in full
    val ARRAYTHRESHOLD = 1000   // longest array to show in full
    val shape = nd.shape
    val space = totalSpace - shape.length
    if (shape.length != 1) {
      val (length, postfix) =
        if (shape.product > THRESHOLD) {
          // reduced NDArray
          (1, s"\n${" " * (space + 1)}... with length ${shape(0)}\n")
        } else {
          (shape(0), "")
        }
      for (num <- 0 until length) {
        val output = buildStringHelper(nd.at(num), totalSpace)
        result += s"$output\n"
      }
      result = s"${" " * space}[\n$result${" " * space}$postfix]"
    } else {
      if (shape(0) > ARRAYTHRESHOLD) {
        // reduced Array
        val front = nd.slice(0, 10)
        val back = nd.slice(shape(0) - 10, shape(0) - 1)
        result = s"${" " * space}[${front.toArray.mkString(",")} ... ${back.toArray.mkString(",")}]"
      } else {
        result = s"${" " * space}[${nd.toArray.mkString(",")}]"
      }
    }
    result
  }
}
