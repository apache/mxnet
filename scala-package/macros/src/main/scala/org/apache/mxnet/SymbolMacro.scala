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
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddSymbolFunctions(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate non-typesafe method for Symbol operations
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolMacro.addDefs
}

private[mxnet] class AddSymbolAPIs(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate typesafe method for Symbol
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) = macro TypedSymbolAPIMacro.typeSafeAPIDefs
}

private[mxnet] class AddSymbolRandomAPIs(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate typesafe method for Random Symbol
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) =
  macro TypedSymbolRandomAPIMacro.typeSafeAPIDefs
}

/**
  * For non-typed Symbol API
  */
private[mxnet] object SymbolMacro extends GeneratorBase {

  /**
    * Methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees Annottees used to define Class or Module
    * @return Generated code for injection
    */
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    impl(c)(isContrib, annottees: _*)
  }

  private def impl(c: blackbox.Context)
                  (isContrib: Boolean, annottees: c.Expr[Any]*): c.Expr[Nothing] = {
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

/**
  * Symbol.api code generation
  */
private[mxnet] object TypedSymbolAPIMacro extends GeneratorBase {

  /**
    * Methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees Annottees used to define Class or Module
    * @return Generated code for injection
    */
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    val functionDefs = typeSafeFunctionsToGenerate(isSymbol = true, isContrib)
      .map(f => buildTypedFunction(c)(f))

    structGeneration(c)(functionDefs, annottees: _*)
  }

  /**
    * Methods that construct the code and build the syntax tree
    * @param c Context used for code gen
    * @param function Case class that store all information of the single function
    * @return Generated syntax tree
    */
  protected def buildTypedFunction(c: blackbox.Context)
                                  (function: Func): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.Symbol"

    // Construct API arguments declaration
    val argDecl = super.typedFunctionCommonArgDef(function) :+
      "name : String = null" :+
      "attr : Map[String, String] = null"

    // Map API input args to backend args
    val backendArgsMapping =
      function.listOfArgs.map { arg =>
        if (arg.argType.equals(s"Array[org.apache.mxnet.Symbol]")) {
          s"args = ${arg.safeArgName}.toSeq"
        } else {
          // all go in kwargs
          if (arg.isOptional) {
            s"""if (!${arg.safeArgName}.isEmpty) map("${arg.argName}") = ${arg.safeArgName}.get"""
          } else {
            s"""map("${arg.argName}") = ${arg.safeArgName}"""
          }
        }
      }

    val impl =
      s"""
         |def ${function.name}
         |  (${argDecl.mkString(",")}): $returnType = {
         |
         |  val map = scala.collection.mutable.Map[String, Any]()
         |  var args = scala.collection.Seq[org.apache.mxnet.Symbol]()
         |
         |  ${backendArgsMapping.mkString("\n")}
         |
         |  org.apache.mxnet.Symbol.createSymbolGeneral(
         |    "${function.name}", name, attr, args, map.toMap)
         |}
       """.stripMargin

    c.parse(impl).asInstanceOf[DefDef]
  }
}


/**
  * Symbol.random code generation
  */
private[mxnet] object TypedSymbolRandomAPIMacro extends GeneratorBase
  with RandomHelpers {

  /**
    * Methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees Annottees used to define Class or Module
    * @return Generated code for injection
    */
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    val functionDefs = typeSafeRandomFunctionsToGenerate(isSymbol = true)
      .map(f => buildTypedFunction(c)(f))

    structGeneration(c)(functionDefs, annottees: _*)
  }

  /**
    * Methods that construct the code and build the syntax tree
    * @param c Context used for code gen
    * @param function Case class that store all information of the single function
    * @return Generated syntax tree
    */
  protected def buildTypedFunction(c: blackbox.Context)
                                  (function: Func): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.Symbol"

    // Construct API arguments declaration
    val argDecl = super.typedFunctionCommonArgDef(function) :+
      "name : String = null" :+
      "attr : Map[String, String] = null"

    // Map API input args to backend args
    val backendArgsMapping =
      function.listOfArgs.map { arg =>
        if (arg.argType.equals(s"Array[org.apache.mxnet.Symbol]")) {
          s"args = ${arg.safeArgName}.toSeq"
        } else {
          // all go in kwargs
          if (arg.isOptional) {
            s"""if (${arg.safeArgName}.isDefined) map("${arg.argName}") = ${arg.safeArgName}.get"""
          } else {
            s"""map("${arg.argName}") = ${arg.safeArgName}"""
          }
        }
      }

    val impl =
      s"""
         |def ${function.name}${randomGenericTypeSpec(true, true)}
         |  (${argDecl.mkString(",")}): $returnType = {
         |
         |  val map = scala.collection.mutable.Map[String, Any]()
         |  var args = scala.collection.Seq[org.apache.mxnet.Symbol]()
         |  val isScalar = SymbolOrScalar[T].isScalar
         |
         |  ${backendArgsMapping.mkString("\n")}
         |
         |  val target = if(isScalar) {
         |    "random_${function.name}"
         |  } else {
         |    "sample_${function.name}"
         |  }
         |
         |  ${unhackNormalFunc(function)}
         |
         |  org.apache.mxnet.Symbol.createSymbolGeneral(
         |    target, name, attr, args, map.toMap)
         |}
       """.stripMargin

    c.parse(impl).asInstanceOf[DefDef]
  }
}

