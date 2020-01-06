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

import org.apache.mxnet.Base._

/**
  * NumpyScope object provides util functions for turning on/off NumPy compatibility
  * and checking whether NumPy compatibility has been turned on/off. NumPy compatibility
  * is introduced first to support zero-dim and zero-size tensors as in NumPy.
  */
object NumpyScope {
  def setNumpyShape(isNpComp: Boolean): Boolean = {
    val prev = new RefInt()
    checkCall(_LIB.mxSetIsNumpyShape(if (isNpComp) 1 else 0, prev))
    if (prev.value != 0) true else false
  }

  def isNumpyShape: Boolean = {
    val curr = new RefInt
    checkCall(_LIB.mxIsNumpyShape(curr))
    if (curr.value != 0) true else false
  }

  def enableNumpyShape: NumpyScope = {
    new NumpyScope(true)
  }


  def disableNumpyShape: NumpyScope = {
    new NumpyScope(false)
  }
}

class NumpyScope(var isCompatible: Boolean) {
  private var prev: Boolean = false

  def withScope[T](body: => T): T = {
    prev = NumpyScope.setNumpyShape(isCompatible)
    try {
      body
    } finally {
      if (prev != isCompatible) {
        NumpyScope.setNumpyShape(prev)
      }
    }
  }
}
