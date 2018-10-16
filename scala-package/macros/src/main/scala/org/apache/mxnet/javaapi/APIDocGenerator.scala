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

import java.io._
import java.security.MessageDigest

import org.apache.mxnet.init.Base._
import org.apache.mxnet.utils.CToScalaUtils

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * This object will generate the Java documentation of the new Java API
  * One file namely: NDArrayBase.scala
  * The code will be executed during Macros stage and file live in Core stage
  */
private[mxnet] object APIDocGenerator{
  case class absClassArg(argName : String, argType : String, argDesc : String, isOptional : Boolean)
  case class absClassFunction(name : String, desc : String,
                           listOfArgs: List[absClassArg], returnType : String)


  def MD5Generator(input : String) : String = {
    val md = MessageDigest.getInstance("MD5")
    md.update(input.getBytes("UTF-8"))
    val digest = md.digest()
    org.apache.commons.codec.binary.Base64.encodeBase64URLSafeString(digest)
  }

  def absClassGen(FILE_PATH : String) : String = {
    // scalastyle:off
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")
    val absClassFunctions = getSymbolNDArrayMethods()
    // TODO: Add Filter to the same location in case of refactor
    val absFuncs = absClassFunctions.filterNot(_.name.startsWith("_"))
      .filterNot(ele => notGenerated.contains(ele.name))
      .map(absClassFunction => {
      val scalaDoc = generateAPIDocFromBackend(absClassFunction)
      val defBody = generateAPISignature(absClassFunction)
      s"$scalaDoc\n$defBody"
    })
    val packageName = "NDArrayBase"
    val apacheLicence = "/*\n* Licensed to the Apache Software Foundation (ASF) under one or more\n* contributor license agreements.  See the NOTICE file distributed with\n* this work for additional information regarding copyright ownership.\n* The ASF licenses this file to You under the Apache License, Version 2.0\n* (the \"License\"); you may not use this file except in compliance with\n* the License.  You may obtain a copy of the License at\n*\n*    http://www.apache.org/licenses/LICENSE-2.0\n*\n* Unless required by applicable law or agreed to in writing, software\n* distributed under the License is distributed on an \"AS IS\" BASIS,\n* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n* See the License for the specific language governing permissions and\n* limitations under the License.\n*/\n"
    val scalaStyle = "// scalastyle:off"
    val packageDef = "package org.apache.mxnet.javaapi"
    val imports = "import org.apache.mxnet.annotation.Experimental"
    val absClassDef = s"abstract class $packageName"
    val finalStr = s"$apacheLicence\n$scalaStyle\n$packageDef\n$imports\n$absClassDef {\n${absFuncs.mkString("\n")}\n}"
    val pw = new PrintWriter(new File(FILE_PATH + s"$packageName.scala"))
    pw.write(finalStr)
    pw.close()
    MD5Generator(finalStr)
  }

  // Generate ScalaDoc type
  def generateAPIDocFromBackend(func : absClassFunction, withParam : Boolean = true) : String = {
    val desc = ArrayBuffer[String]()
    desc += "  * <pre>"
      func.desc.split("\n").foreach({ currStr =>
      desc += s"  * $currStr"
    })
    desc += "  * </pre>"
    val params = func.listOfArgs.map({ absClassArg =>
      val currArgName = absClassArg.argName match {
                case "var" => "vari"
                case "type" => "typeOf"
                case _ => absClassArg.argName
      }
      s"  * @param $currArgName\t\t${absClassArg.argDesc}"
    })
    val returnType = s"  * @return ${func.returnType}"
    if (withParam) {
      s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n$returnType\n  */"
    } else {
      s"  /**\n${desc.mkString("\n")}\n$returnType\n  */"
    }
  }

  def generateAPISignature(func : absClassFunction) : String = {
    var argDef = ListBuffer[String]()
    var classDef = ListBuffer[String]()
    func.listOfArgs.foreach(absClassArg => {
      val currArgName = absClassArg.argName match {
        case "var" => "vari"
        case "type" => "typeOf"
        case _ => absClassArg.argName
      }
      if (absClassArg.isOptional) {
        classDef += s"def set${absClassArg.argName}(${absClassArg.argName} : ${absClassArg.argType}) : ${func.name}BuilderBase"
      }
      else {
        argDef += s"$currArgName : ${absClassArg.argType}"
      }
    })
    classDef += s"def setout(out : NDArray) : ${func.name}BuilderBase"
    classDef += s"def invoke() : org.apache.mxnet.javaapi.NDArrayFuncReturn"
    val experimentalTag = "@Experimental"
    var finalStr = s"$experimentalTag\ndef ${func.name} (${argDef.mkString(", ")}) : ${func.name}BuilderBase\n"
    finalStr += s"abstract class ${func.name}BuilderBase {\n  ${classDef.mkString("\n  ")}\n}"
    finalStr
  }


  // List and add all the atomic symbol functions to current module.
  private def getSymbolNDArrayMethods(): List[absClassFunction] = {
    val opNames = ListBuffer.empty[String]
    val returnType = "NDArray"
    _LIB.mxListAllOpNames(opNames)
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName, "org.apache.mxnet.javaapi." + returnType)
    }).toList.filterNot(_.name.startsWith("_")).groupBy(_.name.toLowerCase).map(ele => {
      // Pattern matching for not generating depreciated method
      if (ele._2.length == 1) ele._2.head
      else {
        if (ele._2.head.name.head.isLower) ele._2.head
        else ele._2.last
      }
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
    val argList = argNames zip argTypes zip argDescs map { case ((argName, argType), argDesc) =>
      val typeAndOption = CToScalaUtils.argumentCleaner(argName, argType, returnType, "javaapi.Shape")
      new absClassArg(argName, typeAndOption._1, argDesc, typeAndOption._2)
    }
    new absClassFunction(aliasName, desc.value, argList.toList, returnType)
  }
}
