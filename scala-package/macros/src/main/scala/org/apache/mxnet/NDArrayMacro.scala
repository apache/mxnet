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
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.typeSafeAPIDefs
}

private[mxnet] class AddNDArrayRandomAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro NDArrayMacro.typeSafeRandomAPIDefs
}


private[mxnet] object NDArrayMacro extends GeneratorBase {
  type NDArrayArg = Arg
  type NDArrayFunction = Func

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(annottees: _*)
  }

  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typedAPIImpl(c)(annottees: _*)
  }

  def typeSafeRandomAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typedRandomAPIImpl(c)(annottees: _*)
  }

  // scalastyle:off havetype

  private val ndarrayFunctions = buildFunctionList(false)

  private val rndFunctions = buildRandomFunctionList(false)

  /**
    * Implementation for fixed input API structure
    */
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

    structGeneration(c)(functionDefs, annottees: _*)
  }

  /**
    * Implementation for Dynamic typed API NDArray.api.<functioname>
    */
  private def typedAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")

    val newNDArrayFunctions = {
      if (isContrib) ndarrayFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else ndarrayFunctions.filterNot(f => f.name.startsWith("_"))
    }.filterNot(ele => notGenerated.contains(ele.name))

    val functionDefs = newNDArrayFunctions.map(f => buildTypedFunction(c)(f))

    structGeneration(c)(functionDefs, annottees: _*)
  }

  /**
    * Implementation for Dynamic typed API NDArray.random.<functioname>
    */
  private def typedRandomAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    val functionDefs = rndFunctions.map(f => buildRandomTypedFunction(c)(f))
    structGeneration(c)(functionDefs, annottees: _*)
  }

  private def buildTypedFunction(c: blackbox.Context)
                                (function: NDArrayFunction): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"
    val arrayType = "org.apache.mxnet.NDArray"

    // Construct argument field
    val argDef = ListBuffer[String]()
    argDef ++= buildArgDefs(function)
    argDef += "out : Option[NDArray] = None"

    // Construct Implementation field
    var impl = ListBuffer[String]()
    impl += "val map = scala.collection.mutable.Map[String, Any]()"
    impl += "val args = scala.collection.mutable.ArrayBuffer.empty[NDArray]"

    // NDArray arg implementation
    impl ++=
      function.listOfArgs.map { ndarrayarg =>
        // TODO: Currently we do not add place holder for NDArray
        // Example: an NDArray operator like the following format
        // nd.foo(arg1: NDArray(required), arg2: NDArray(Optional), arg3: NDArray(Optional)
        // If we place nd.foo(arg1, arg3 = arg3), do we need to add place holder for arg2?
        // What it should be?
        val base =
          if (ndarrayarg.argType.equals(arrayType)) {
            s"args += ${ndarrayarg.safeArgName}"
          } else if (ndarrayarg.argType.equals(s"Array[$arrayType]")) {
            s"args ++= ${ndarrayarg.safeArgName}"
          } else {
            s"""map("${ndarrayarg.argName}") = ${ndarrayarg.safeArgName}"""
          }
        if (ndarrayarg.isOptional) s"if (!${ndarrayarg.safeArgName}.isEmpty) $base.get"
        else base
      }

    impl += "if (!out.isEmpty) map(\"out\") = out.get"
    impl += "org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(" +
      s""""${function.name}", args.toSeq, map.toMap)"""

    // Combine and build the function string
    var finalStr = s"def ${function.name}"
    finalStr += s" (${argDef.mkString(",")}) : $returnType"
    finalStr += s" = {${impl.mkString("\n")}}"
    c.parse(finalStr).asInstanceOf[DefDef]
  }

  private def buildRandomTypedFunction(c: blackbox.Context)
                                      (function: NDArrayFunction): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"
    val arrayType = "org.apache.mxnet.NDArray"

    // Construct argument field
    val argDef = ListBuffer[String]()
    argDef ++= buildArgDefs(function)
    argDef += "out : Option[NDArray] = None"

    // Construct Implementation field
    var impl = ListBuffer[String]()
    impl += "val map = scala.collection.mutable.Map[String, Any]()"
    impl += "val args = scala.collection.mutable.ArrayBuffer.empty[NDArray]"

    // determine what target to call
    val arg = function.listOfArgs.filter(arg => arg.argType == "Any").head
    if(arg.isOptional) {
      impl +=
        s"""val target = ${arg.safeArgName} match {
           |   case Some(a:$arrayType) => "sample_${function.name}"
           |   case None => "sample_${function.name}"
           |   case _ => "random_${function.name}"
           |}
      """.stripMargin
    } else {
      impl +=
        s"""val target = ${arg.safeArgName} match {
           |   case _:$arrayType => "sample_${function.name}"
           |   case _ => "random_${function.name}"
           |}
      """.stripMargin
    }

    // NDArray arg implementation
    impl ++=
      function.listOfArgs.map { ndarrayarg =>
        // no Array[] in random/sample module, but let's keep that for a future case
        val base =
          if (ndarrayarg.argType.equals(arrayType)) {
            s"args += ${ndarrayarg.safeArgName}"
          } else if (ndarrayarg.argType.equals(s"Array[$arrayType]")) {
            s"args ++= ${ndarrayarg.safeArgName}"
          } else {
            s"""map("${ndarrayarg.argName}") = ${ndarrayarg.safeArgName}"""
          }
        if (ndarrayarg.isOptional) s"if (!${ndarrayarg.safeArgName}.isEmpty) $base.get"
        else base
      }

    impl += "if (!out.isEmpty) map(\"out\") = out.get"
    impl += "org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(" +
      s"target, args.toSeq, map.toMap)"

    // Combine and build the function string
    var finalStr = s"def ${function.name}"
    finalStr += s" (${argDef.mkString(",")}) : $returnType"
    finalStr += s" = {${impl.mkString("\n")}}"
    c.parse(finalStr).asInstanceOf[DefDef]
  }

}
