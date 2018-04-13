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

import scala.collection.mutable.{HashMap, ListBuffer}
import org.apache.mxnet.init.Base._

private[mxnet] class AddSymbolBaseFunctions() {
  private[mxnet] def addDocs() = SymbolDocMacros.addDefs
}

private[mxnet] object SymbolDocMacros {

  case class SymbolFunction(handle: SymbolHandle, paramStr: String)

  def addDefs() : Unit = {
    val baseDir = System.getProperty("user.dir")
    val targetDir = baseDir + "/core/src/main/scala/org/apache/mxnet/"
    SEImpl(targetDir)
  }

  def SEImpl(FILE_PATH : String) : Unit = {
    var symbolFunctions: List[SymbolFunction] = initSymbolModule()
    import java.io._
    val pw = new PrintWriter(new File(FILE_PATH))
    // scalastyle:off
    pw.write("/*\n* Licensed to the Apache Software Foundation (ASF) under one or more\n* contributor license agreements.  See the NOTICE file distributed with\n* this work for additional information regarding copyright ownership.\n* The ASF licenses this file to You under the Apache License, Version 2.0\n* (the \"License\"); you may not use this file except in compliance with\n* the License.  You may obtain a copy of the License at\n*\n*    http://www.apache.org/licenses/LICENSE-2.0\n*\n* Unless required by applicable law or agreed to in writing, software\n* distributed under the License is distributed on an \"AS IS\" BASIS,\n* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n* See the License for the specific language governing permissions and\n* limitations under the License.\n*/\n\npackage org.apache.mxnet\n")
    // scalastyle:on
    pw.write(s"trait SymbolBase {\n\n")
    pw.write(s"  // scalastyle:off\n")
    symbolFunctions = symbolFunctions.distinct
    for (ele <- symbolFunctions) {
      val temp = ele.paramStr + "\n\n"
      pw.write(temp)
    }
    pw.write(s"\n\n}")
    pw.close()
  }


  /*
    Code copies from the SymbolMacros Class
   */
  private def initSymbolModule(): List[SymbolFunction] = {
    var opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    opNames = opNames.distinct
    val result : ListBuffer[SymbolFunction] = ListBuffer[SymbolFunction]()
    opNames.foreach(opName => {
      val opHandle = new RefLong
      // printf(opName)
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName, result)
    })

    result.toList
  }

  private def makeAtomicSymbolFunction(handle: SymbolHandle,
                                       aliasName: String, result : ListBuffer[SymbolFunction])
  : Unit = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val returnType = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs, returnType)

    if (name.value.charAt(0) == '_') {
      // Internal function
    } else {
      val paramStr =
        traitgen(name.value, desc.value, argNames, argTypes, argDescs, returnType.value)
      val extraDoc: String = if (keyVarNumArgs.value != null && keyVarNumArgs.value.length > 0) {
        s"This function support variable length of positional input (${keyVarNumArgs.value})."
      } else {
        ""
      }
      result +=  SymbolFunction(handle, paramStr)
    }
  }


  def traitgen(functionName : String,
               functionDesc : String,
               argNames : Seq[String],
               argTypes : Seq[String],
               argDescs : Seq[String],
               returnType : String) : String = {
    val desc = functionDesc.split("\n") map { currStr =>
      s"  * $currStr"
    }
    val params =
      (argNames zip argTypes zip argDescs) map { case ((argName, argType), argDesc) =>
        // val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"  * @param $argName\t\t$argDesc"
      }
    val traitsec =
      (argNames zip argTypes) map { case ((argName, argType)) =>
        val currArgType = CodeClean.cleanUp(argType)
        var currArgName = ""
        if (argName.equals("var")) {
          currArgName = "vari"
        } else {
          currArgName = argName
        }
        s"$currArgName : $currArgType"
      }
    // scalastyle:off
    val defaultConfig = s"(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol"
    // s"/**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n* @return $returnType\n*/\ndef $functionName(${traitsec.mkString(", ")}) : Any"
    s"  /**\n${desc.mkString("\n")}\n${params.mkString("\n")}\n  * @return $returnType\n  */\n  def $functionName$defaultConfig"
    // scalastyle:on
  }
}

private[mxnet] object CodeClean {


  val typeMap : HashMap[String, String] = HashMap(
    ("Shape(tuple)", "Shape"),
    ("Symbol", "Symbol"),
    ("NDArray", "Symbol"),
    ("NDArray-or-Symbol", "Symbol"),
    // TODO: Add def
    ("Symbol[]", "Any"),
    ("NDArray[]", "Any"),
    ("NDArray-or-Symbol[]", "Any"),
    ("int(non-negative)", "Any"),
    ("long(non-negative)", "Any"),
    ("ShapeorNone", "Option[Shape]"),
    ("real_t", "Any"), // MXFloat
    ("float", "Any"),
    ("intorNone", "Option[Int]"),
    ("SymbolorSymbol[]", "Any"),
    ("tupleof<float>", "Any"),
    // End Missing section
    ("int", "Int"),
    ("long", "Long"),
    ("double", "Double"),
    ("string", "String"),
    ("boolean", "Boolean")
  )


  def conversion(in : String, optional : String) : String = {
    val out = in match {
      // deal with []
      case "Shape" => "new Shape()"
      // deal with '6000' => 6000
      case "Int" | "Option[Int]" | "Option[Shape]" => optional.replaceAll("'", "")
      // deal with string default
      case "String" => optional.replaceAll("'", "\"")
      // Deal with Boolean
      case "Boolean" => {
        if (optional.charAt(0) == '0') {
          "false"
        } else {
          "true"
        }
      }
      // Anything else
      case _ => optional
    }

    out
  }

  def cleanUp(in : String) : String = {
    val spaceRemoved = in.replaceAll("\\s+", "")
    var commaRemoved : Array[String] = new Array[String](0)
    // Deal with the case e.g: stype : {'csr', 'default', 'row_sparse'}
    if (spaceRemoved.charAt(0)== '{') {
      val endIdx = spaceRemoved.indexOf('}')
      commaRemoved = spaceRemoved.substring(endIdx + 1).split(",")
      // commaRemoved(0) = spaceRemoved.substring(0, endIdx+1)
      commaRemoved(0) = "string"
    } else {
      commaRemoved = spaceRemoved.split(",")
    }
    var typeConv = ""
    var optionalField = ""
    // println("Try to find key " + commaRemoved(0))
    if (commaRemoved.length < 1) {
      printf("Empty Field Generated\n")
    } else if (commaRemoved.length == 3) {

      // Something to do with Optional
      typeConv = typeMap(commaRemoved(0))
      optionalField = " = " + conversion(typeConv, commaRemoved(2).split("=")(1))
    } else if (commaRemoved.length > 3) {
      // TODO: Field over 3, need to rework
      typeConv = "Any"
      printf("Field Over 3, please reformat %s", in)
    } else {
      typeConv = typeMap(commaRemoved(0))
    }
    //    if (!typeMap.contains(commaRemoved(0))) {
    //      logger.error("First Field not recognized " + commaRemoved(0))
    //    } else {
    //      typeConv = typeMap(commaRemoved(0))
    //    }
    val out = typeConv + optionalField
    out
  }

}