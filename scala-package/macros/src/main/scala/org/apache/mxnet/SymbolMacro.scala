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

private[mxnet] class AddSymbolFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolMacro.addDefs
}

private[mxnet] class AddSymbolAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro TypedSymbolAPIMacro.typeSafeAPIDefs
}

private[mxnet] object SymbolMacro extends GeneratorBase {

  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    impl(c)(isContrib, annottees: _*)
  }

  private def impl(c: blackbox.Context)
                  (isContrib: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val functions = functionsToGenerate(isSymbol = false, isContrib)

    val functionDefs = functions.map { symbolfunction =>
      val funcName = symbolfunction.name
      val tName = TermName(funcName)
      q"""
            def $tName(name : String = null, attr : Map[String, String] = null)
              (args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null)
              : org.apache.mxnet.Symbol = {
                createSymbolGeneral($funcName,name,attr,args,kwargs)
            }
         """.asInstanceOf[DefDef]
    }

    structGeneration(c)(functionDefs, annottees: _*)
  }
}

private[mxnet] object TypedSymbolAPIMacro extends GeneratorBase {

  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    val functions = typeSafeFunctionsToGenerate(isSymbol = true, isContrib)

    val functionDefs = functions.map(f => buildTypedFunction(c)(f))
    structGeneration(c)(functionDefs, annottees: _*)
  }

  protected def buildTypedFunction(c: blackbox.Context)
                                  (function: Func): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.Symbol"
    val symbolType = "org.apache.mxnet.Symbol"

    // Construct argument field
    val argDef = ListBuffer[String]()
    argDef ++= typedFunctionCommonArgDef(function)
    argDef += "name : String = null"
    argDef += "attr : Map[String, String] = null"

    // Construct Implementation field
    val impl = ListBuffer[String]()
    impl += "val map = scala.collection.mutable.Map[String, Any]()"
    impl += s"var args = scala.collection.Seq[$symbolType]()"

    // Symbol arg implementation
    impl ++= function.listOfArgs.map { arg =>
      if (arg.argType.equals(s"Array[$symbolType]")) {
        s"if (!${arg.safeArgName}.isEmpty) args = ${arg.safeArgName}.toSeq"
      } else {
        // all go in kwargs
        if (arg.isOptional) {
          s"""if (!${arg.safeArgName}.isEmpty) map("${arg.argName}") = ${arg.safeArgName}.get"""
        } else {
          s"""map("${arg.argName}") = ${arg.safeArgName}"""
        }
      }
    }

    impl +=
      s"""org.apache.mxnet.Symbol.createSymbolGeneral(
         |  "${function.name}", name, attr, args, map.toMap)
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
