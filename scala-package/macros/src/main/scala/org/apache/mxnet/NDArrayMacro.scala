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

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddNDArrayFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.addDefs
}

private[mxnet] class AddNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro TypedNDArrayAPIMacro.typeSafeAPIDefs
}

private[mxnet] object NDArrayMacro extends GeneratorBase {

  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    impl(c)(isContrib, annottees: _*)
  }

  private def impl(c: blackbox.Context)
                  (isContrib: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val functions = functionsToGenerate(isSymbol = false, isContrib)

    val functionDefs = functions.flatMap { NDArrayfunction =>
      val funcName = NDArrayfunction.name
      val termName = TermName(funcName)
      Seq(
        // e.g def transpose(kwargs: Map[String, Any] = null)(args: Any*)
        q"""
             def $termName(kwargs: Map[String, Any] = null)(args: Any*) = {
               genericNDArrayFunctionInvoke($funcName, args, kwargs)
             }
          """.asInstanceOf[DefDef],
        // e.g def transpose(args: Any*)
        q"""
             def $termName(args: Any*) = {
               genericNDArrayFunctionInvoke($funcName, args, null)
             }
          """.asInstanceOf[DefDef]
      )
    }

    structGeneration(c)(functionDefs, annottees: _*)
  }
}

private[mxnet] object TypedNDArrayAPIMacro extends GeneratorBase {

  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    val functions = typeSafeFunctionsToGenerate(isSymbol = false, isContrib)

    val functionDefs = functions.map(f => buildTypedFunction(c)(f))
    structGeneration(c)(functionDefs, annottees: _*)
  }

  protected def buildTypedFunction(c: blackbox.Context)
                                  (function: Func): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"
    val ndarrayType = "org.apache.mxnet.NDArray"

    // Construct argument field
    val argDef = ListBuffer[String]()
    argDef ++= typedFunctionCommonArgDef(function)
    argDef += "out : Option[NDArray] = None"

    // Construct Implementation field
    var impl = ListBuffer[String]()
    impl += "val map = scala.collection.mutable.Map[String, Any]()"
    impl += s"val args = scala.collection.mutable.ArrayBuffer.empty[$ndarrayType]"

    // NDArray arg implementation
    impl ++= function.listOfArgs.map { arg =>
      if (arg.argType.equals(s"Array[$ndarrayType]")) {
        s"args ++= ${arg.safeArgName}"
      } else {
        val base =
          if (arg.argType.equals(ndarrayType)) {
            // ndarrays go to args
            s"args += ${arg.safeArgName}"
          } else {
            // other types go to kwargs
            s"""map("${arg.argName}") = ${arg.safeArgName}"""
          }
        if (arg.isOptional) s"if (!${arg.safeArgName}.isEmpty) $base.get"
        else base
      }
    }

    impl +=
      s"""if (!out.isEmpty) map("out") = out.get
         |org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(
         |  "${function.name}", args.toSeq, map.toMap)
       """.stripMargin

    // Combine and build the function string
    val finalStr =
      s"""def ${function.name}
         |   (${argDef.mkString(",")}) : $returnType
         | = {${impl.mkString("\n")}}
       """.stripMargin

    c.parse(finalStr).asInstanceOf[DefDef]
  }
}
