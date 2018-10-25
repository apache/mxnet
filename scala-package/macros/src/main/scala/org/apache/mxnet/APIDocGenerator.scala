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

import java.io._
import java.security.MessageDigest

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * This object will generate the Scala documentation of the new Scala API
  * Two file namely: SymbolAPIBase.scala and NDArrayAPIBase.scala
  * The code will be executed during Macros stage and file live in Core stage
  */
private[mxnet] object APIDocGenerator extends GeneratorBase {
  type absClassArg = Arg
  type absClassFunction = Func

  def main(args: Array[String]): Unit = {
    val FILE_PATH = args(0)
    val hashCollector = ListBuffer[String]()
    hashCollector += absClassGen(FILE_PATH, true)
    hashCollector += absClassGen(FILE_PATH, false)
    hashCollector += absRndClassGen(FILE_PATH, true)
    hashCollector += absRndClassGen(FILE_PATH, false)
    hashCollector += nonTypeSafeClassGen(FILE_PATH, true)
    hashCollector += nonTypeSafeClassGen(FILE_PATH, false)
    val finalHash = hashCollector.mkString("\n")
  }

  def MD5Generator(input: String): String = {
    val md = MessageDigest.getInstance("MD5")
    md.update(input.getBytes("UTF-8"))
    val digest = md.digest()
    org.apache.commons.codec.binary.Base64.encodeBase64URLSafeString(digest)
  }

  def absRndClassGen(FILE_PATH: String, isSymbol: Boolean): String = {
    val funcs = getSymbolNDArrayMethods(isSymbol)
      .filter(f => f.name.startsWith("_sample_") || f.name.startsWith("_random_"))
      .map(f => f.copy(name = f.name.stripPrefix("_")))
    val body = funcs.map(func => {
      val scalaDoc = generateAPIDocFromBackend(func)
      val decl = generateRandomAPISignature(func, isSymbol)
      s"$scalaDoc\n$decl"
    })
    writeFile(
      FILE_PATH,
      if (isSymbol) "SymbolRandomAPIBase" else "NDArrayRandomAPIBase",
      body)
  }

  def absClassGen(FILE_PATH: String, isSymbol: Boolean): String = {
    val notGenerated = Set("Custom")
    val funcs = getSymbolNDArrayMethods(isSymbol)
      .filterNot(_.name.startsWith("_"))
      .filterNot(ele => notGenerated.contains(ele.name))
    val body = funcs.map(func => {
      val scalaDoc = generateAPIDocFromBackend(func)
      val decl = generateAPISignature(func, isSymbol)
      s"$scalaDoc\n$decl"
    })
    writeFile(
      FILE_PATH,
      if (isSymbol) "SymbolAPIBase" else "NDArrayAPIBase",
      body)
  }

  def nonTypeSafeClassGen(FILE_PATH: String, isSymbol: Boolean): String = {
    val absClassFunctions = getSymbolNDArrayMethods(isSymbol)
    val absFuncs = absClassFunctions
      .filterNot(_.name.startsWith("_"))
      .map(absClassFunction => {
        val scalaDoc = generateAPIDocFromBackend(absClassFunction, false)
        if (isSymbol) {
          val defBody =
            s"def ${absClassFunction.name}(name : String = null, attr : Map[String, String] = null)" +
              s"(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): " +
              s"org.apache.mxnet.Symbol"
          s"$scalaDoc\n$defBody"
        } else {
          val defBodyWithKwargs = s"def ${absClassFunction.name}(kwargs: Map[String, Any] = null)" +
            s"(args: Any*): " +
            s"org.apache.mxnet.NDArrayFuncReturn"
          val defBody = s"def ${absClassFunction.name}(args: Any*): " +
            s"org.apache.mxnet.NDArrayFuncReturn"
          s"$scalaDoc\n$defBodyWithKwargs\n$scalaDoc\n$defBody"
        }
      })
    val packageName = if (isSymbol) "SymbolBase" else "NDArrayBase"
    writeFile(FILE_PATH, packageName, absFuncs)
  }

  def writeFile(FILE_PATH: String, packageName: String, body: Seq[String]): String = {
    val apacheLicence =
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
    val packageDef = "package org.apache.mxnet"
    val imports = "import org.apache.mxnet.annotation.Experimental"
    val absClassDef = s"abstract class $packageName"
    val finalStr =
      s"""$apacheLicence
         |$scalaStyle
         |$packageDef
         |$imports
         |$absClassDef {
         |${body.mkString("\n")}
         |}""".stripMargin
    val pw = new PrintWriter(new File(FILE_PATH + s"$packageName.scala"))
    pw.write(finalStr)
    pw.close()
    MD5Generator(finalStr)
  }

  // Generate ScalaDoc type
  def generateAPIDocFromBackend(func: absClassFunction, withParam: Boolean = true): String = {
    val desc = ArrayBuffer[String]()
    desc += "  * <pre>"
    func.desc.split("\n").foreach({ currStr =>
      desc += s"  * $currStr"
    })
    desc += "  * </pre>"
    val params = func.listOfArgs.map({ absClassArg =>
      s"  * @param ${absClassArg.safeArgName}\t\t${absClassArg.argDesc}"
    })
    val returnType = s"  * @return ${func.returnType}"
    if (withParam) {
      s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n$returnType\n  */"
    } else {
      s"  /**\n${desc.mkString("\n")}\n$returnType\n  */"
    }
  }

  def generateRandomAPISignature(func: absClassFunction, isSymbol: Boolean): String = {
    generateAPISignature(func, isSymbol)
  }

  def generateAPISignature(func: absClassFunction, isSymbol: Boolean): String = {
    val argDef = ListBuffer[String]()

    argDef ++= buildArgDefs(func)

    if (isSymbol) {
      argDef += "name : String = null"
      argDef += "attr : Map[String, String] = null"
    } else {
      argDef += "out : Option[NDArray] = None"
    }

    val returnType = func.returnType

    val experimentalTag = "@Experimental"
    s"$experimentalTag\ndef ${func.name} (${argDef.mkString(", ")}) : $returnType"
  }

  // List and add all the atomic symbol functions to current module.
  private def getSymbolNDArrayMethods(isSymbol: Boolean): List[absClassFunction] = {
    buildFunctionList(isSymbol)
  }

}
