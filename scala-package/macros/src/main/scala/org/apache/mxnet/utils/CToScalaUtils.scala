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
package org.apache.mxnet.utils

private[mxnet] object CToScalaUtils {

  private val javaType = Map(
    "float" -> "java.lang.Float",
    "int" -> "java.lang.Integer",
    "long" -> "java.lang.Long",
    "double" -> "java.lang.Double",
    "bool" -> "java.lang.Boolean")
  private val scalaType = Map(
    "float" -> "Float",
    "int" -> "Int",
    "long" -> "Long",
    "double" -> "Double",
    "bool" -> "Boolean")

  /**
    * Convert C++ Types to Scala Types
    * @param in Input raw string that contains C type docs
    * @param argType Arg type that used for error messaging
    * @param argName Arg name used for error messaging
    * @param returnType The type that NDArray/Symbol should be
    * @param isJava Check if generating for Java
    * @return String that contains right Scala/Java types
    */
  def typeConversion(in : String, argType : String = "", argName : String,
                     returnType : String, isJava : Boolean) : String = {
    val header = returnType.split("\\.").dropRight(1)
    val types = if (isJava) javaType else scalaType
    in match {
      case "Shape(tuple)" | "ShapeorNone" => s"${header.mkString(".")}.Shape"
      case "Symbol" | "NDArray" | "NDArray-or-Symbol" => returnType
      case "Symbol[]" | "NDArray[]" | "NDArray-or-Symbol[]" | "SymbolorSymbol[]"
      => s"Array[$returnType]"
      case "float" | "real_t" | "floatorNone" => types("float")
      case "int" | "intorNone" | "int(non-negative)" => types("int")
      case "long" | "long(non-negative)" => types("long")
      case "double" | "doubleorNone" => types("double")
      case "string" => "String"
      case "boolean" | "booleanorNone" => types("bool")
      case "tupleof<int>"| "tupleof<long>" | "tupleof<float>" | "tupleof<double>" |
           "tupleof<intorNone>" | "tupleof<tupleof<int>>" | "tupleof<Shape(tuple)>" |
           "tupleof<>" | "ptr" | "" => "Any"
      case default => throw new IllegalArgumentException(
        s"Invalid type for args: $default\nString argType: $argType\nargName: $argName")
    }
  }


  /**
    * By default, the argType come from the C++ API is a description more than a single word
    * For Example:
    *   <C++ Type>, <Required/Optional>, <Default=>
    * The three field shown above do not usually come at the same time
    * This function used the above format to determine if the argument is
    * optional, what is it Scala type and possibly pass in a default value
    * @param argName The name of the argument
    * @param argType Raw arguement Type description
    * @param returnType Return type of the function (Symbol/NDArray)
    * @param isJava Check if Java args should be generated
    * @return (Scala_Type, isOptional)
    */
  def argumentCleaner(argName: String, argType : String,
                      returnType : String, isJava : Boolean) : (String, Boolean) = {
    val spaceRemoved = argType.replaceAll("\\s+", "")
    var commaRemoved : Array[String] = new Array[String](0)
    // Deal with the case e.g: stype : {'csr', 'default', 'row_sparse'}
    if (spaceRemoved.charAt(0)== '{') {
      val endIdx = spaceRemoved.indexOf('}')
      commaRemoved = spaceRemoved.substring(endIdx + 1).split(",")
      commaRemoved(0) = "string"
    } else {
      commaRemoved = spaceRemoved.split(",")
    }
    // Optional Field
    if (commaRemoved.length >= 3) {
      // arg: Type, optional, default = Null
      require(commaRemoved(1).equals("optional"),
        s"""expected "optional" got ${commaRemoved(1)}""")
      require(commaRemoved(2).startsWith("default="),
        s"""expected "default=..." got ${commaRemoved(2)}""")
      (typeConversion(commaRemoved(0), argType, argName, returnType, isJava), true)
    } else if (commaRemoved.length == 2 || commaRemoved.length == 1) {
      val tempType = typeConversion(commaRemoved(0), argType, argName, returnType, isJava)
      val tempOptional = tempType.equals("org.apache.mxnet.Symbol")
      (tempType, tempOptional)
    } else {
      throw new IllegalArgumentException(
        s"Unrecognized arg field: $argType, ${commaRemoved.length}")
    }

  }
}
