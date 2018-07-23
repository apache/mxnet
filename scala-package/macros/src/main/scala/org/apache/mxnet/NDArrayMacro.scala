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
import org.apache.mxnet.utils.{CToScalaUtils, OperatorBuildUtils}

import scala.annotation.StaticAnnotation
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddNDArrayFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.addDefs
}

private[mxnet] class AddNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.typeSafeAPIDefs
}

private[mxnet] object NDArrayMacro {
  case class NDArrayArg(argName: String, argType: String, isOptional : Boolean)
  case class NDArrayFunction(name: String, listOfArgs: List[NDArrayArg])

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(annottees: _*)
  }
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typeSafeAPIImpl(c)(annottees: _*)
  }
  // scalastyle:off havetype

  private val ndarrayFunctions: List[NDArrayFunction] = initNDArrayModule()

  private def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(_.name.startsWith("_contrib_"))
      else ndarrayFunctions.filterNot(_.name.startsWith("_"))
    }

     val functionDefs = newNDArrayFunctions flatMap { NDArrayfunction =>
        val funcName = NDArrayfunction.name
        val termName = TermName(funcName)
       Seq(
            // scalastyle:off
            // (yizhi) We are investigating a way to make these functions type-safe
            // and waiting to see the new approach is stable enough.
            // Thus these functions may be deprecated in the future.
            // e.g def transpose(kwargs: Map[String, Any] = null)(args: Any*)
            q"def $termName(kwargs: Map[String, Any] = null)(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, kwargs)}".asInstanceOf[DefDef],
            // e.g def transpose(args: Any*)
            q"def $termName(args: Any*) = {genericNDArrayFunctionInvoke($funcName, args, null)}".asInstanceOf[DefDef]
            // scalastyle:on
          )
        }

    structGeneration(c)(functionDefs, annottees : _*)
  }

  private def typeSafeAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else ndarrayFunctions.filterNot(_.name.startsWith("_"))
    }.filterNot(ele => notGenerated.contains(ele.name))

    val functionDefs = newNDArrayFunctions.map { ndarrayfunction =>

      // Construct argument field
      var argDef = ListBuffer[String]()
      // Construct Implementation field
      var impl = ListBuffer[String]()
      impl += "val map = scala.collection.mutable.Map[String, Any]()"
      impl += "val args = scala.collection.mutable.ArrayBuffer.empty[NDArray]"
      ndarrayfunction.listOfArgs.foreach({ ndarrayarg =>
        // var is a special word used to define variable in Scala,
        // need to changed to something else in order to make it work
        val currArgName = ndarrayarg.argName match {
          case "var" => "vari"
          case "type" => "typeOf"
          case default => ndarrayarg.argName
        }
        if (ndarrayarg.isOptional) {
          argDef += s"${currArgName} : Option[${ndarrayarg.argType}] = None"
        }
        else {
          argDef += s"${currArgName} : ${ndarrayarg.argType}"
        }
        // NDArray arg implementation
        val returnType = "org.apache.mxnet.NDArray"

        // TODO: Currently we do not add place holder for NDArray
        // Example: an NDArray operator like the following format
        // nd.foo(arg1: NDArray(required), arg2: NDArray(Optional), arg3: NDArray(Optional)
        // If we place nd.foo(arg1, arg3 = arg3), do we need to add place holder for arg2?
        // What it should be?
        val base =
          if (ndarrayarg.argType.equals(returnType)) {
            s"args += $currArgName"
          } else if (ndarrayarg.argType.equals(s"Array[$returnType]")){
            s"args ++= $currArgName"
          } else {
            "map(\"" + ndarrayarg.argName + "\") = " + currArgName
          }
        impl.append(
          if (ndarrayarg.isOptional) s"if (!$currArgName.isEmpty) $base.get"
          else base
        )
      })
      // add default out parameter
      argDef += "out : Option[NDArray] = None"
      impl += "if (!out.isEmpty) map(\"out\") = out.get"
      // scalastyle:off
      impl += "org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(\"" + ndarrayfunction.name + "\", args.toSeq, map.toMap)"
      // scalastyle:on
      // Combine and build the function string
      val returnType = "org.apache.mxnet.NDArrayFuncReturn"
      var finalStr = s"def ${ndarrayfunction.name}"
      finalStr += s" (${argDef.mkString(",")}) : $returnType"
      finalStr += s" = {${impl.mkString("\n")}}"
      c.parse(finalStr).asInstanceOf[DefDef]
    }

    structGeneration(c)(functionDefs, annottees : _*)
  }

  private def structGeneration(c: blackbox.Context)
                              (funcDef : List[c.universe.DefDef], annottees: c.Expr[Any]*)
  : c.Expr[Any] = {
    import c.universe._
    val inputs = annottees.map(_.tree).toList
    // pattern match on the inputs
    val modDefs = inputs map {
      case ClassDef(mods, name, something, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ funcDef)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ModuleDef(mods, name, q)
      case ex =>
        throw new IllegalArgumentException(s"Invalid macro input: $ex")
    }
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
    val paramStr = OperatorBuildUtils.ctypes2docstring(argNames, argTypes, argDescs)
    val extraDoc: String = if (keyVarNumArgs.value != null && keyVarNumArgs.value.length > 0) {
      s"This function support variable length of positional input (${keyVarNumArgs.value})."
    } else {
      ""
    }
    val realName = if (aliasName == name.value) "" else s"(a.k.a., ${name.value})"
    val docStr = s"$aliasName $realName\n${desc.value}\n\n$paramStr\n$extraDoc\n"
    // scalastyle:off println
    if (System.getenv("MXNET4J_PRINT_OP_DEF") != null
      && System.getenv("MXNET4J_PRINT_OP_DEF").toLowerCase == "true") {
      println("NDArray function definition:\n" + docStr)
    }
    // scalastyle:on println
    val argList = argNames zip argTypes map { case (argName, argType) =>
      val typeAndOption =
        CToScalaUtils.argumentCleaner(argName, argType, "org.apache.mxnet.NDArray")
      new NDArrayArg(argName, typeAndOption._1, typeAndOption._2)
    }
    new NDArrayFunction(aliasName, argList.toList)
  }
}
