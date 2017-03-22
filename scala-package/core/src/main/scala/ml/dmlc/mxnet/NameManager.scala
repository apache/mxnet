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

package ml.dmlc.mxnet

import scala.collection.mutable

/**
 * NameManager to do automatic naming.
 * User can also inherit this object to change naming behavior.
 */
class NameManager {
  val counter: mutable.Map[String, Int] = mutable.HashMap.empty[String, Int]
  /**
   * Get the canonical name for a symbol.
   * This is default implementation.
   * When user specified a name,
   * the user specified name will be used.
   * When user did not, we will automatically generate a name based on hint string.
   *
   * @param name : The name user specified.
   * @param hint : A hint string, which can be used to generate name.
   * @return A canonical name for the user.
   */
  def get(name: Option[String], hint: String): String = {
    name.getOrElse {
      if (!counter.contains(hint)) {
        counter(hint) = 0
      }
      val generatedName = s"$hint${counter(hint)}"
      counter(hint) += 1
      generatedName
    }
  }

  def withScope[T](body: => T): T = {
    val oldManager = NameManager.current
    NameManager.setCurrentManager(this)
    try {
      body
    } finally {
      NameManager.setCurrentManager(oldManager)
    }
  }
}

object NameManager {
  private var _current = new NameManager()
  def current: NameManager = _current
  private def setCurrentManager(manager: NameManager): Unit = {
    _current = manager
  }
}
