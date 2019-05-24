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

import scala.language.implicitConversions

object MX_PRIMITIVES {

  /**
    * This defines the basic primitives we can use in Scala for mathematical
    * computations in NDArrays.This gives us a flexibility to expand to
    * more supported primitives in the future. Currently Float and Double
    * are supported. The functions which accept MX_PRIMITIVE_TYPE as input can also accept
    * plain old Float and Double data as inputs because of the underlying
    * implicit conversion between primitives to MX_PRIMITIVE_TYPE.
    */
  trait MX_PRIMITIVE_TYPE extends Ordered[MX_PRIMITIVE_TYPE]{

    def toString: String

    def unary_- : MX_PRIMITIVE_TYPE
  }

  trait MXPrimitiveOrdering extends Ordering[MX_PRIMITIVE_TYPE] {

    def compare(x: MX_PRIMITIVE_TYPE, y: MX_PRIMITIVE_TYPE): Int = x.compare(y)

  }

  implicit object MX_PRIMITIVE_TYPE extends MXPrimitiveOrdering

  /**
    * Wrapper over Float in Scala.
    * @param data
    */
  class MX_FLOAT(val data: Float) extends MX_PRIMITIVE_TYPE {

    override def toString: String = data.toString

    override def unary_- : MX_PRIMITIVE_TYPE = new MX_FLOAT(data.unary_-)

    override def compare(that: MX_PRIMITIVE_TYPE): Int = {
      this.data.compareTo(that.asInstanceOf[MX_FLOAT].data)
    }
  }

  implicit def FloatToMX_Float(d : Float): MX_FLOAT = new MX_FLOAT(d)

  implicit def MX_FloatToFloat(d: MX_FLOAT) : Float = d.data

  implicit def IntToMX_Float(d: Int): MX_FLOAT = new MX_FLOAT(d.toFloat)

  /**
    * Wrapper over Double in Scala.
    * @param data
    */
  class MX_Double(val data: Double) extends MX_PRIMITIVE_TYPE {

    override def toString: String = data.toString

    override def unary_- : MX_PRIMITIVE_TYPE = new MX_Double(data.unary_-)

    override def compare(that: MX_PRIMITIVE_TYPE): Int = {
      this.data.compareTo(that.asInstanceOf[MX_Double].data)
    }
  }

  implicit def DoubleToMX_Double(d : Double): MX_Double = new MX_Double(d)

  implicit def MX_DoubleToDouble(d: MX_Double) : Double = d.data

  def isValidMxPrimitiveType(num : Any) : Boolean = {
    num match {
      case valid @ (_: Float | _: Double) => true
      case _ => false
    }
  }
}
