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
import org.apache.mxnet.utils.CToScalaUtils
import java.io._
import java.security.MessageDigest

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

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
    val hashCollector = ListBuffer[String]()
    hashCollector += absClassGen(FILE_PATH, true)
    hashCollector += absClassGen(FILE_PATH, false)
    hashCollector += nonTypeSafeClassGen(FILE_PATH, true)
    hashCollector += nonTypeSafeClassGen(FILE_PATH, false)
    // Generate Java API documentation
    hashCollector += javaClassGen(FILE_PATH + "javaapi/")
    val finalHash = hashCollector.mkString("\n")
  }

  def MD5Generator(input : String) : String = {
    val md = MessageDigest.getInstance("MD5")
    md.update(input.getBytes("UTF-8"))
    val digest = md.digest()
    org.apache.commons.codec.binary.Base64.encodeBase64URLSafeString(digest)
  }

  def fileGen(filePath : String, packageName : String, packageDef : String,
              absFuncs : List[String]) : String = {
    val apacheLicense =
      """/*
        |* Licensed to the Apache Software Foundation (ASF) under one or more
        |* contributor license agreements.  See the NOTICE file distributed with
        |* this work for additional information regarding copyright ownership.
        |* The ASF licenses this file to You under the Apache License, Version 2.0
        |* (the "License"); you may not use this file except in compliance with
        |* the License.  You may obtain a copy of the License at
        |*
        |*    http://www.apache.org/licenses/LICENSE-2.0
        |*
        |* Unless required by applicable law or agreed to in writing, software
        |* distributed under the License is distributed on an "AS IS" BASIS,
        |* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        |* See the License for the specific language governing permissions and
        |* limitations under the License.
        |*/
        |""".stripMargin
    val scalaStyle = "// scalastyle:off"
    val imports = "import org.apache.mxnet.annotation.Experimental"
    val absClassDef = s"abstract class $packageName"

    val finalStr =
      s"""$apacheLicense
         |$scalaStyle
         |$packageDef
         |$imports
         |$absClassDef {
         |${absFuncs.mkString("\n")}
         |}""".stripMargin
    val pw = new PrintWriter(new File(filePath + s"$packageName.scala"))
    pw.write(finalStr)
    pw.close()
    MD5Generator(finalStr)
  }

  def absClassGen(filePath : String, isSymbol : Boolean) : String = {
    val absClassFunctions = getSymbolNDArrayMethods(isSymbol)
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")
    // TODO: Add Filter to the same location in case of refactor
    val absFuncs = absClassFunctions.filterNot(_.name.startsWith("_"))
      .filterNot(ele => notGenerated.contains(ele.name))
      .map(absClassFunction => {
      val scalaDoc = generateAPIDocFromBackend(absClassFunction)
      val defBody = generateAPISignature(absClassFunction, isSymbol)
      s"$scalaDoc\n$defBody"
    })
    val packageName = if (isSymbol) "SymbolAPIBase" else "NDArrayAPIBase"
    val packageDef = "package org.apache.mxnet"
    fileGen(filePath, packageName, packageDef, absFuncs)
  }

  def javaClassGen(filePath : String) : String = {
    val notGenerated = Set("Custom")
    val absClassFunctions = getSymbolNDArrayMethods(false, true)
    // TODO: Add Filter to the same location in case of refactor
    val absFuncs = absClassFunctions.filterNot(_.name.startsWith("_"))
      .filterNot(ele => notGenerated.contains(ele.name))
      .map(absClassFunction => {
        val scalaDoc = generateAPIDocFromBackend(absClassFunction)
        val defBody = generateJavaAPISignature(absClassFunction)
        s"$scalaDoc\n$defBody"
      })
    val packageName = "NDArrayBase"
    val packageDef = "package org.apache.mxnet.javaapi"
    fileGen(filePath, packageName, packageDef, absFuncs)
  }

  def nonTypeSafeClassGen(filePath : String, isSymbol : Boolean) : String = {
    // scalastyle:off
    val absClassFunctions = getSymbolNDArrayMethods(isSymbol)
    val absFuncs = absClassFunctions.map(absClassFunction => {
      val scalaDoc = generateAPIDocFromBackend(absClassFunction, false)
      if (isSymbol) {
        val defBody = s"def ${absClassFunction.name}(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol"
        s"$scalaDoc\n$defBody"
      } else {
        val defBodyWithKwargs = s"def ${absClassFunction.name}(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn"
        val defBody = s"def ${absClassFunction.name}(args: Any*) : org.apache.mxnet.NDArrayFuncReturn"
        s"$scalaDoc\n$defBodyWithKwargs\n$scalaDoc\n$defBody"
      }
    })
    val packageName = if (isSymbol) "SymbolBase" else "NDArrayBase"
    val packageDef = "package org.apache.mxnet"
    fileGen(filePath, packageName, packageDef, absFuncs)
  }

  /**
    * Some of the C++ type name is not valid in Scala
    * such as var and type. This method is to convert
    * them into other names to get it passed
    * @param in the input String
    * @return converted name string
    */
  def safetyNameCheck(in : String) : String = {
    in match {
      case "var" => "vari"
      case "type" => "typeOf"
      case _ => in
    }
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
      val currArgName = safetyNameCheck(absClassArg.argName)
      s"  * @param $currArgName\t\t${absClassArg.argDesc}"
    })
    val returnType = s"  * @return ${func.returnType}"
    if (withParam) {
      s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n$returnType\n  */"
    } else {
      s"  /**\n${desc.mkString("\n")}\n$returnType\n  */"
    }
  }

  def generateAPISignature(func : absClassFunction, isSymbol : Boolean) : String = {
    var argDef = ListBuffer[String]()
    func.listOfArgs.foreach(absClassArg => {
      val currArgName = safetyNameCheck(absClassArg.argName)
      if (absClassArg.isOptional) {
        argDef += s"$currArgName : Option[${absClassArg.argType}] = None"
      }
      else {
        argDef += s"$currArgName : ${absClassArg.argType}"
      }
    })
    var returnType = func.returnType
    if (isSymbol) {
      argDef += "name : String = null"
      argDef += "attr : Map[String, String] = null"
    } else {
      argDef += "out : Option[NDArray] = None"
      returnType = "org.apache.mxnet.NDArrayFuncReturn"
    }
    val experimentalTag = "@Experimental"
    s"$experimentalTag\ndef ${func.name} (${argDef.mkString(", ")}) : $returnType"
  }

  def generateJavaAPISignature(func : absClassFunction) : String = {
    val useParamObject = func.listOfArgs.count(arg => arg.isOptional) >= 2
    var argDef = ListBuffer[String]()
    var classDef = ListBuffer[String]()
    func.listOfArgs.foreach(absClassArg => {
      val currArgName = safetyNameCheck(absClassArg.argName)
      // scalastyle:off
      if (absClassArg.isOptional && useParamObject) {
        classDef +=
          s"""private var $currArgName: ${absClassArg.argType} = null
             |def set${currArgName.capitalize}($currArgName : ${absClassArg.argType}): ${func.name}Param = {
             |  this.$currArgName = $currArgName
             |  this
             | }""".stripMargin
      }
      else {
        argDef += s"$currArgName : ${absClassArg.argType}"
      }
      classDef += s"def get${currArgName.capitalize}() = this.$currArgName"
      // scalastyle:on
    })
    val experimentalTag = "@Experimental"
    val returnType = "Array[NDArray]"
    if(useParamObject) {
      classDef +=
        s"""private var out : org.apache.mxnet.NDArray = null
           |def setOut(out : NDArray) : ${func.name}Param = {
           |  this.out = out
           |  this
           | }
           | def getOut() = this.out
           | """.stripMargin
      s"""$experimentalTag
          | def ${func.name}(po: ${func.name}Param) : $returnType
          | class ${func.name}Param(${argDef.mkString(",")}) {
          |  ${classDef.mkString("\n  ")}
          | }""".stripMargin
    } else {
      argDef += "out : NDArray"
      s"""$experimentalTag
         | def ${func.name}(${argDef.mkString(", ")}) : $returnType
         | """.stripMargin
    }
  }


  // List and add all the atomic symbol functions to current module.
  private def getSymbolNDArrayMethods(isSymbol : Boolean,
                                      isJava : Boolean = false): List[absClassFunction] = {
    val opNames = ListBuffer.empty[String]
    val returnType = if (isSymbol) "Symbol" else "NDArray"
    val returnHeader = if (isJava) "org.apache.mxnet.javaapi." else "org.apache.mxnet."
    _LIB.mxListAllOpNames(opNames)
    // TODO: Add '_linalg_', '_sparse_', '_image_' support
    // TODO: Add Filter to the same location in case of refactor
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName, returnHeader + returnType)
    }).filterNot(_.name.startsWith("_")).groupBy(_.name.toLowerCase).map(ele => {
      // Pattern matching for not generating depreciated method
      if (ele._2.length == 1) ele._2.head
      else {
        if (ele._2.head.name.head.isLower) ele._2.head
        else ele._2.last
      }
    }).toList
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle,
                                       aliasName: String, returnType : String)
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
      val typeAndOption = CToScalaUtils.argumentCleaner(argName, argType, returnType)
      new absClassArg(argName, typeAndOption._1, argDesc, typeAndOption._2)
    }
    new absClassFunction(aliasName, desc.value, argList.toList, returnType)
  }
}
