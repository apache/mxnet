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

import org.apache.mxnet.init.Base._
import org.apache.mxnet.utils.CToScalaUtils

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddJNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro JavaNDArrayMacro.typeSafeAPIDefs
}

private[mxnet] object JavaNDArrayMacro {
  case class NDArrayArg(argName: String, argType: String, isOptional : Boolean)
  case class NDArrayFunction(name: String, listOfArgs: List[NDArrayArg])

  // scalastyle:off havetype
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typeSafeAPIImpl(c)(annottees: _*)
  }
  // scalastyle:off havetype

  private val ndarrayFunctions: List[NDArrayFunction] = initNDArrayModule()

  private def typeSafeAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddJNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else ndarrayFunctions.filterNot(_.name.startsWith("_"))
    }.filterNot(ele => notGenerated.contains(ele.name)).groupBy(_.name.toLowerCase).map(ele => {
      // Pattern matching for not generating depreciated method
      if (ele._2.length == 1) ele._2.head
      else {
        if (ele._2.head.name.head.isLower) ele._2.head
        else ele._2.last
      }
    })

    val functionDefs = ListBuffer[DefDef]()
    val classDefs = ListBuffer[ClassDef]()

    newNDArrayFunctions.foreach { ndarrayfunction =>

      val useParamObject = ndarrayfunction.listOfArgs.count(arg => arg.isOptional) >= 2
      val header = if (useParamObject) "this." else ""
      // Construct argument field with all required args
      var argDef = ListBuffer[String]()
      // Construct Optional Arg
      var OptionArgDef = ListBuffer[String]()
      // Construct function Implementation field (e.g norm)
      var impl = ListBuffer[String]()
      impl += "val map = scala.collection.mutable.Map[String, Any]()"
      impl +=
        "val args= scala.collection.mutable.ArrayBuffer.empty[org.apache.mxnet.NDArray]"
      // Construct Class Implementation (e.g normBuilder)
      var classImpl = ListBuffer[String]()
      ndarrayfunction.listOfArgs.foreach({ ndarrayArg =>
        // var is a special word used to define variable in Scala,
        // need to changed to something else in order to make it work
        var currArgName = ndarrayArg.argName match {
          case "var" => "vari"
          case "type" => "typeOf"
          case _ => ndarrayArg.argName
        }
        if (ndarrayArg.isOptional && useParamObject) {
          OptionArgDef += s"private var $currArgName : ${ndarrayArg.argType} = null"
          val tempDef = s"def set${currArgName.capitalize}($currArgName : ${ndarrayArg.argType})"
          val tempImpl = s"this.$currArgName = $currArgName\nthis"
          classImpl += s"$tempDef = {$tempImpl}"
        } else {
          argDef += s"$currArgName : ${ndarrayArg.argType}"
        }
        // NDArray arg implementation
        val returnType = "org.apache.mxnet.javaapi.NDArray"
        val base =
          if (ndarrayArg.argType.equals(returnType)) {
            s"args += $header$currArgName"
          } else if (ndarrayArg.argType.equals(s"Array[$returnType]")){
            s"$header$currArgName.foreach(args+=_)"
          } else {
            "map(\"" + ndarrayArg.argName + "\") = " + header + currArgName
          }
        impl.append(
          if (ndarrayArg.isOptional) s"if ($header$currArgName != null) $base"
          else base
        )
      })
      // add default out parameter
      if (useParamObject) {
        classImpl +=
          "def setOut(out : org.apache.mxnet.javaapi.NDArray) = {this.out = out\nthis}"
      } else {
        argDef += s"out: org.apache.mxnet.javaapi.NDArray"
      }
      impl += "if (" + header + "out != null) map(\"out\") = " + header + "out"
      OptionArgDef += "var out : org.apache.mxnet.NDArray = null"
      val returnType = "org.apache.mxnet.javaapi.NDArrayFuncReturn"
      // scalastyle:off
      // Combine and build the function string
      impl += "org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(\"" +
        ndarrayfunction.name + "\", args.toSeq, map.toMap)"
      if (useParamObject) {
        val classDef =
          s"""class ${ndarrayfunction.name}Param(${argDef.mkString(",")})
             | extends ${ndarrayfunction.name}ParamBase(${argDef.mkString(",")}) {
             |   ${OptionArgDef.mkString("\n")}
             |   ${classImpl.mkString("\n")}
             |   def invoke() : $returnType = {
             |     ${impl.mkString("\n")}
             |   }
             | }""".stripMargin
        classDefs += c.parse(classDef).asInstanceOf[ClassDef]
        val funcDef =
          s"""def ${ndarrayfunction.name}(po: ${ndarrayfunction.name}ParamBase): $returnType = {
             |  po.invoke()
             | }""".stripMargin
        functionDefs += c.parse(funcDef).asInstanceOf[DefDef]
      } else {
        val funcDef =
          s"""def ${ndarrayfunction.name}(${argDef.mkString(",")}): $returnType = {
             |  ${impl.mkString("\n")}
             | }""".stripMargin
        functionDefs += c.parse(funcDef).asInstanceOf[DefDef]
      }
    }

    structGeneration(c)(functionDefs.toList, classDefs.toList, annottees : _*)
  }

  private def structGeneration(c: blackbox.Context)
                              (funcDef : List[c.universe.DefDef],
                               classDef : List[c.universe.ClassDef],
                               annottees: c.Expr[Any]*)
  : c.Expr[Any] = {
    import c.universe._
    val inputs = annottees.map(_.tree).toList
    // pattern match on the inputs
    var modDefs = inputs map {
      case ClassDef(mods, name, something, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef ++ classDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef ++ classDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ModuleDef(mods, name, q)
      case ex =>
        throw new IllegalArgumentException(s"Invalid macro input: $ex")
    }
    //    modDefs ++= classDef
    // wrap the result up in an Expr, and return it
    val result = c.Expr(Block(modDefs, Literal(Constant())))
    result
  }

  // List and add all the atomic symbol functions to current module.
  private def initNDArrayModule(): List[NDArrayFunction] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeNDArrayFunction(opHandle.value, opName)
    }).toList
  }

  // Create an atomic symbol function by handle and function name.
  private def makeNDArrayFunction(handle: NDArrayHandle, aliasName: String)
  : NDArrayFunction = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs)
    val argList = argNames zip argTypes map { case (argName, argType) =>
      val typeAndOption =
        CToScalaUtils.argumentCleaner(argName, argType,
          "org.apache.mxnet.javaapi.NDArray")
      new NDArrayArg(argName, typeAndOption._1, typeAndOption._2)
    }
    new NDArrayFunction(aliasName, argList.toList)
  }
}
