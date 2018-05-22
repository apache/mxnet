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

/**
  * This object will generate the Scala documentation of the new Scala API
  * Two file namely: SymbolAPIBase.scala and NDArrayAPIBase.scala
  * The code will be executed during Macros stage and file live in Core stage
  */
private[mxnet] object APIDocGenerator{
  case class absClassArg(argName : String, argType : String, argDesc : String, isOptional : Boolean)
  case class absClassFunction(name : String, desc : String,
                           listOfArgs: List[absClassArg], returnType : String)


  def main(args: Array[String]) : Unit = {
    val FILE_PATH = args(0)
    absClassGen(FILE_PATH, true)
    absClassGen(FILE_PATH, false)
  }

  def absClassGen(FILE_PATH : String, isSymbol : Boolean) : Unit = {
    // scalastyle:off
    val absClassFunctions = getSymbolNDArrayMethods(isSymbol)
    val absFuncs = absClassFunctions.filterNot(_.name.startsWith("_")).map(absClassFunction => {
      val scalaDoc = generateAPIDocFromBackend(absClassFunction)
      val defBody = generateAPISignature(absClassFunction, isSymbol)
      s"$scalaDoc\n$defBody"
    })
    val packageName = if (isSymbol) "SymbolAPIBase" else "NDArrayAPIBase"
    val apacheLicence = "/*\n* Licensed to the Apache Software Foundation (ASF) under one or more\n* contributor license agreements.  See the NOTICE file distributed with\n* this work for additional information regarding copyright ownership.\n* The ASF licenses this file to You under the Apache License, Version 2.0\n* (the \"License\"); you may not use this file except in compliance with\n* the License.  You may obtain a copy of the License at\n*\n*    http://www.apache.org/licenses/LICENSE-2.0\n*\n* Unless required by applicable law or agreed to in writing, software\n* distributed under the License is distributed on an \"AS IS\" BASIS,\n* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n* See the License for the specific language governing permissions and\n* limitations under the License.\n*/\n"
    val scalaStyle = "// scalastyle:off"
    val packageDef = "package org.apache.mxnet"
    val absClassDef = s"abstract class $packageName"
    val finalStr = s"$apacheLicence\n$scalaStyle\n$packageDef\n$absClassDef {\n${absFuncs.mkString("\n")}\n}"
    import java.io._
    val pw = new PrintWriter(new File(FILE_PATH + s"$packageName.scala"))
    pw.write(finalStr)
    pw.close()
  }

  // Generate ScalaDoc type
  def generateAPIDocFromBackend(traitFunc : absClassFunction) : String = {
    val desc = traitFunc.desc.split("\n").map({ currStr =>
      s"  * $currStr"
    })
    val params = traitFunc.listOfArgs.map({ absClassArg =>
      val currArgName = absClassArg.argName match {
                case "var" => "vari"
                case "type" => "typeOf"
                case _ => absClassArg.argName
      }
      s"  * @param $currArgName\t\t${absClassArg.argDesc}"
    })
    val returnType = s"  * @return ${traitFunc.returnType}"
    s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n$returnType\n  */"
  }

  def generateAPISignature(absClassFunc : absClassFunction, isSymbol : Boolean) : String = {
    var argDef = ListBuffer[String]()
    absClassFunc.listOfArgs.foreach(absClassArg => {
      val currArgName = absClassArg.argName match {
        case "var" => "vari"
        case "type" => "typeOf"
        case _ => absClassArg.argName
      }
      if (absClassArg.isOptional) {
        argDef += s"$currArgName : Option[${absClassArg.argType}] = None"
      }
      else {
        argDef += s"$currArgName : ${absClassArg.argType}"
      }
    })
    if (isSymbol) {
      argDef += "name : String = null"
      argDef += "attr : Map[String, String] = null"
    }
    s"def ${absClassFunc.name} (${argDef.mkString(", ")}) : ${absClassFunc.returnType}"
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
  private def getSymbolNDArrayMethods(isSymbol : Boolean): List[absClassFunction] = {
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
  : absClassFunction = {
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
      new absClassArg(argName, typeAndOption._1, argDesc, typeAndOption._2)
    }
    new absClassFunction(aliasName, desc.value, argList.toList, returnType)
  }
}
