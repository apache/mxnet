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

private[mxnet] class AddNDArrayFunctions(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate non-typesafe method for NDArray operations
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.addDefs
}

private[mxnet] class AddNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate typesafe method for NDArray operations
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) = macro TypedNDArrayAPIMacro.typeSafeAPIDefs
}

private[mxnet] class AddNDArrayRandomAPIs(isContrib: Boolean) extends StaticAnnotation {
/**
  * Generate typesafe method for Random Symbol
  * @param annottees Annottees used to define Class or Module
  * @return Generated code for injection
  */
  private[mxnet] def macroTransform(annottees: Any*) =
  macro TypedNDArrayRandomAPIMacro.typeSafeAPIDefs
}

/**
  * For non-typed NDArray API
  */
private[mxnet] object NDArrayMacro extends GeneratorBase {
  /**
    * Methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees Annottees used to define Class or Module
    * @return Generated code for injection
    */
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    impl(c)(isContrib, annottees: _*)
  }

  private def impl(c: blackbox.Context)
                  (isContrib: Boolean, annottees: c.Expr[Any]*): c.Expr[Nothing] = {
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

/**
  * NDArray.api code generation
  */
private[mxnet] object TypedNDArrayAPIMacro extends GeneratorBase {
  /**
    * Methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees Annottees used to define Class or Module
    * @return Generated code for injection
    */
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    val functionDefs = typeSafeFunctionsToGenerate(isSymbol = false, isContrib)
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

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"

    // Construct API arguments declaration
    val argDecl = super.typedFunctionCommonArgDef(function) :+ "out : Option[NDArray] = None"

    // Map API input args to backend args
    val backendArgsMapping =
      function.listOfArgs.map { arg =>
        // ndarrays go to args, other types go to kwargs
        if (arg.argType.equals(s"Array[org.apache.mxnet.NDArray]")) {
          s"args ++= ${arg.safeArgName}.toSeq"
        } else {
          val base = if (arg.argType.equals("org.apache.mxnet.NDArray")) {
            s"args += ${arg.safeArgName}"
          } else {
            s"""map("${arg.argName}") = ${arg.safeArgName}"""
          }
          if (arg.isOptional) s"if (!${arg.safeArgName}.isEmpty) $base.get"
          else base
        }
      }

    val impl =
      s"""
         |def ${function.name}
         |  (${argDecl.mkString(",")}): $returnType = {
         |
         |  val map = scala.collection.mutable.Map[String, Any]()
         |  val args = scala.collection.mutable.ArrayBuffer.empty[org.apache.mxnet.NDArray]
         |
         |  if (!out.isEmpty) map("out") = out.get
         |
         |  ${backendArgsMapping.mkString("\n")}
         |
         |  org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(
         |    "${function.name}", args.toSeq, map.toMap)
         |}
       """.stripMargin

    c.parse(impl).asInstanceOf[DefDef]
  }
}


/**
  * NDArray.random code generation
  */
private[mxnet] object TypedNDArrayRandomAPIMacro extends GeneratorBase
  with RandomHelpers {
  /**
    * methods that check the ``isContrib`` and call code generation
    * @param c Context used for code gen
    * @param annottees annottees used to define Class or Module
    * @return generated code for injection
    */
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    // Note: no contrib managed in this module

    val functionDefs = typeSafeRandomFunctionsToGenerate(isSymbol = false)
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

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"

    // Construct API arguments declaration
    val argDecl = super.typedFunctionCommonArgDef(function) :+ "out : Option[NDArray] = None"

    // Map API input args to backend args
    val backendArgsMapping =
      function.listOfArgs.map { arg =>
        // ndarrays go to args, other types go to kwargs
        if (arg.argType.equals("Array[org.apache.mxnet.NDArray]")) {
          s"args ++= ${arg.safeArgName}.toSeq"
        } else {
          if (arg.argType.equals("T")) {
            if (arg.isOptional) {
              s"""if(${arg.safeArgName}.isDefined) {
                 |  if(isScalar) {
                 |    map("${arg.argName}") = ${arg.safeArgName}.get
                 |  } else {
                 |    args += ${arg.safeArgName}.get.asInstanceOf[org.apache.mxnet.NDArray]
                 |  }
                 |}
             """.stripMargin
            } else {
              s"""if(isScalar) {
                 |  map("${arg.argName}") = ${arg.safeArgName}
                 |} else {
                 |  args += ${arg.safeArgName}.asInstanceOf[org.apache.mxnet.NDArray]
                 |}
             """.stripMargin
            }
          } else {
            if (arg.isOptional) {
              s"""if (${arg.safeArgName}.isDefined) map("${arg.argName}")=${arg.safeArgName}.get"""
            } else {
              s"""map("${arg.argName}") = ${arg.safeArgName}"""
            }
          }
        }
      }

    val impl =
      s"""
         |def ${function.name}${randomGenericTypeSpec(false, true)}
         |  (${argDecl.mkString(",")}): $returnType = {
         |
         |  val map = scala.collection.mutable.Map[String, Any]()
         |  val args = scala.collection.mutable.ArrayBuffer.empty[org.apache.mxnet.NDArray]
         |  val isScalar = NDArrayOrScalar[T].isScalar
         |
         |  if(out.isDefined) map("out") = out.get
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
         |  org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(
         |    target, args.toSeq, map.toMap)
         |}
       """.stripMargin

    c.parse(impl).asInstanceOf[DefDef]
  }


}
