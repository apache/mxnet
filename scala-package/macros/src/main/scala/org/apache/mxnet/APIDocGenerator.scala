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

import org.apache.mxnet.init.Base._

import scala.collection.mutable.ListBuffer

private[mxnet] object APIDocGenerator{
  case class traitArg(argName : String, argType : String, argDesc : String, isOptional : Boolean)
  case class traitFunction(name : String, desc : String,
                           listOfArgs: List[traitArg], returnType : String)


  def main(args: Array[String]) : Unit = {
    val FILE_PATH = args(0)
    traitGen(FILE_PATH)
  }

  def traitGen(FILE_PATH : String) : Unit = {
    // scalastyle:off
    val traitFunctions = initSymbolModule(true)
    val traitfuncs = traitFunctions.filterNot(_.name.startsWith("_")).map(traitfunction => {
      val scalaDoc = ScalaDocGen(traitfunction)
      val traitBody = traitBodyGen(traitfunction)
      s"$scalaDoc\n$traitBody"
    })
    val apacheLicence = "/*\n* Licensed to the Apache Software Foundation (ASF) under one or more\n* contributor license agreements.  See the NOTICE file distributed with\n* this work for additional information regarding copyright ownership.\n* The ASF licenses this file to You under the Apache License, Version 2.0\n* (the \"License\"); you may not use this file except in compliance with\n* the License.  You may obtain a copy of the License at\n*\n*    http://www.apache.org/licenses/LICENSE-2.0\n*\n* Unless required by applicable law or agreed to in writing, software\n* distributed under the License is distributed on an \"AS IS\" BASIS,\n* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n* See the License for the specific language governing permissions and\n* limitations under the License.\n*/\n"
    val scalaStyle = "// scalastyle:off"
    val packageDef = "package org.apache.mxnet"
    val absClassDef = "abstract class SymbolAPIBase"
    val finalStr = s"$apacheLicence\n$scalaStyle\n$packageDef\n$absClassDef {\n${traitfuncs.mkString("\n")}\n}"
    import java.io._
    val pw = new PrintWriter(new File(FILE_PATH + "SymbolAPIBase.scala"))
    pw.write(finalStr)
    pw.close()
  }

  // Generate ScalaDoc type
  def ScalaDocGen(traitFunc : traitFunction) : String = {
    val desc = traitFunc.desc.split("\n").map({ currStr =>
      s"  * $currStr"
    })
    val params = traitFunc.listOfArgs.map({ traitarg =>
      val currArgName = traitarg.argName match {
                case "var" => "vari"
                case "type" => "typeOf"
                case _ => traitarg.argName
      }
      s"  * @param $currArgName\t\t${traitarg.argDesc}"
    })
    val returnType = s"  * @return ${traitFunc.returnType}"
    s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n$returnType\n  */"
  }

  def traitBodyGen(traitFunc : traitFunction) : String = {
    var argDef = ListBuffer[String]()
    traitFunc.listOfArgs.foreach(traitarg => {
              val currArgName = traitarg.argName match {
                case "var" => "vari"
                case "type" => "typeOf"
                case _ => traitarg.argName
              }
              if (traitarg.isOptional) {
                  argDef += s"$currArgName : Option[${traitarg.argType}] = None"
                }
              else {
                  argDef += s"$currArgName : ${traitarg.argType}"
                }
            })
          argDef += "name : String = null"
          argDef += "attr : Map[String, String] = null"
    s"def ${traitFunc.name} (${argDef.mkString(", ")}) : ${traitFunc.returnType}"
  }

  // Convert C++ Types to Scala Types
  def typeConversion(in : String, argType : String = "", returnType : String) : String = {
    in match {
      case "Shape(tuple)" | "ShapeorNone" => "org.apache.mxnet.Shape"
      case "Symbol" | "NDArray" | "NDArray-or-Symbol" => returnType
      case "Symbol[]" | "NDArray[]" | "NDArray-or-Symbol[]" | "SymbolorSymbol[]"
      => s"Array[$returnType]"
      case "float" | "real_t" | "floatorNone" => "org.apache.mxnet.Base.MXFloat"
      case "int" | "intorNone" | "int(non-negative)" => "Int"
      case "long" | "long(non-negative)" => "Long"
      case "double" | "doubleorNone" => "Double"
      case "string" => "String"
      case "boolean" => "Boolean"
      case "tupleof<float>" | "tupleof<double>" | "ptr" | "" => "Any"
      case default => throw new IllegalArgumentException(
        s"Invalid type for args: $default, $argType")
    }
  }


  /**
    * By default, the argType come from the C++ API is a description more than a single word
    * For Example:
    *   <C++ Type>, <Required/Optional>, <Default=>
    * The three field shown above do not usually come at the same time
    * This function used the above format to determine if the argument is
    * optional, what is it Scala type and possibly pass in a default value
    * @param argType Raw arguement Type description
    * @return (Scala_Type, isOptional)
    */
  def argumentCleaner(argType : String, returnType : String) : (String, Boolean) = {
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
      require(commaRemoved(1).equals("optional"))
      require(commaRemoved(2).startsWith("default="))
      (typeConversion(commaRemoved(0), argType, returnType), true)
    } else if (commaRemoved.length == 2 || commaRemoved.length == 1) {
      val tempType = typeConversion(commaRemoved(0), argType, returnType)
      val tempOptional = tempType.equals("org.apache.mxnet.Symbol")
      (tempType, tempOptional)
    } else {
      throw new IllegalArgumentException(
        s"Unrecognized arg field: $argType, ${commaRemoved.length}")
    }

  }


  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(isSymbol : Boolean): List[traitFunction] = {
    val opNames = ListBuffer.empty[String]
    val returnType = if (isSymbol) "Symbol" else "NDArray"
    _LIB.mxListAllOpNames(opNames)
    // TODO: Add '_linalg_', '_sparse_', '_image_' support
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName, "org.apache.mxnet." + returnType)
    }).toList
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle, aliasName: String, returnType : String)
  : traitFunction = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs)

    val realName = if (aliasName == name.value) "" else s"(a.k.a., ${name.value})"

    val argList = argNames zip argTypes zip argDescs map { case ((argName, argType), argDesc) =>
      val typeAndOption = argumentCleaner(argType, returnType)
      new traitArg(argName, typeAndOption._1, argDesc, typeAndOption._2)
    }
    new traitFunction(aliasName, desc.value, argList.toList, returnType)
  }
}
