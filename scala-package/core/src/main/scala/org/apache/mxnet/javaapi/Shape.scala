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

package org.apache.mxnet.javaapi

import collection.JavaConverters._
import scala.language.implicitConversions

/**
  * Shape of [[NDArray]] or other data
  */

class Shape private[mxnet] (val shape: org.apache.mxnet.Shape) {
  def this(dims: java.util.List[java.lang.Integer])
    = this(new org.apache.mxnet.Shape(dims.asScala.map(Int.unbox)))
  def this(dims: Array[Int]) = this(new org.apache.mxnet.Shape(dims))

  def apply(dim: Int): Int = shape.apply(dim)
  def get(dim: Int): Int = apply(dim)
  def size: Int = shape.size
  def length: Int = shape.length
  def drop(dim: Int): Shape = shape.drop(dim)
  def slice(from: Int, end: Int): Shape = shape.slice(from, end)
  def product: Int = shape.product
  def head: Int = shape.head

  def toArray: Array[Int] = shape.toArray
  def toVector: java.util.List[Int] = shape.toVector.asJava

  override def toString(): String = shape.toString
  override def equals(o: Any): Boolean = shape.equals(o)
  override def hashCode(): Int = shape.hashCode()
}

object Shape {
  implicit def fromShape(shape: org.apache.mxnet.Shape): Shape = new Shape(shape)

  implicit def toShape(jShape: Shape): org.apache.mxnet.Shape = jShape.shape
}
