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

/**
 * Attribute manager for scoping.
 * User can also inherit this object to change naming behavior.
 */
private[mxnet] class AttrScope(attr: Map[String, String] = Map.empty) {
  private var _attr = attr
  /**
   * Get the attribute dict given the attribute set by the symbol.
   * @param userDefinedAttr The attribute passed in by user during symbol creation.
   * @return Updated attributes to add other scope related attributes.
   */
  def get(userDefinedAttr: Option[Map[String, String]]): Map[String, String] = {
    _attr ++ userDefinedAttr.getOrElse(Map.empty[String, String])
  }

  def withScope[T](body: => T): T = {
    val oldAttrScope = AttrScope.current
    this._attr = AttrScope.current._attr ++ this._attr
    AttrScope.setCurrentAttr(this)
    try {
      body
    } finally {
      AttrScope.setCurrentAttr(oldAttrScope)
    }
  }
}

private[mxnet] object AttrScope {
  private var _current = new AttrScope()
  def current: AttrScope = _current
  private def setCurrentAttr(attr: AttrScope): Unit = {
    _current = attr
  }

  def apply(attr: Map[String, String] = Map.empty): AttrScope = new AttrScope(attr)
}
