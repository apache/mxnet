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



  // Convert C++ Types to Scala Types
  def typeConversion(in : String, argType : String = "", argName : String,
                     returnType : String) : String = {
    val header = returnType.split("\\.").dropRight(1)
    in match {
      case "Shape(tuple)" | "ShapeorNone" => s"${header.mkString(".")}.Shape"
      case "Symbol" | "NDArray" | "NDArray-or-Symbol" => returnType
      case "Symbol[]" | "NDArray[]" | "NDArray-or-Symbol[]" | "SymbolorSymbol[]"
      => s"Array[$returnType]"
      case "float" | "real_t" | "floatorNone" => "java.lang.Float"
      case "int" | "intorNone" | "int(non-negative)" => "java.lang.Integer"
      case "long" | "long(non-negative)" => "java.lang.Long"
      case "double" | "doubleorNone" => "java.lang.Double"
      case "string" => "String"
      case "boolean" | "booleanorNone" => "java.lang.Boolean"
      case "tupleof<float>" | "tupleof<double>" | "tupleof<>" | "ptr" | "" => "Any"
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
    * @return (Scala_Type, isOptional)
    */
  def argumentCleaner(argName: String, argType : String,
                      returnType : String) : (String, Boolean) = {
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
      (typeConversion(commaRemoved(0), argType, argName, returnType), true)
    } else if (commaRemoved.length == 2 || commaRemoved.length == 1) {
      val tempType = typeConversion(commaRemoved(0), argType, argName, returnType)
      val tempOptional = tempType.equals("org.apache.mxnet.Symbol")
      (tempType, tempOptional)
    } else {
      throw new IllegalArgumentException(
        s"Unrecognized arg field: $argType, ${commaRemoved.length}")
    }

  }
}
