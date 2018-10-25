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
    typeSafeAPIImpl(c)(annottees: _*)
  }
  def typeSafeRandomAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typeSafeRandomAPIImpl(c)(annottees: _*)
  }
  // scalastyle:off havetype

  private val ndarrayFunctions: List[NDArrayFunction] = buildFunctionList(false)

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

    structGeneration(c)(functionDefs, annottees : _*)
  }

  /**
    * Implementation for Dynamic typed API NDArray.random.<functioname>
    */
  private def typeSafeRandomAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val rndFunctions = ndarrayFunctions
      .filter(f => f.name.startsWith("_sample_") || f.name.startsWith("_random_"))
      .map(f => f.copy(name = f.name.stripPrefix("_")))

    val functionDefs = rndFunctions.map(f => buildTypeSafeFunction(c)(f))
    structGeneration(c)(functionDefs, annottees: _*)
  }

  /**
    * Implementation for Dynamic typed API NDArray.api.<functioname>
    */
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
      else ndarrayFunctions.filterNot(f => f.name.startsWith("_"))
    }.filterNot(ele => notGenerated.contains(ele.name))

    val functionDefs = newNDArrayFunctions.map(f => buildTypeSafeFunction(c)(f))

    structGeneration(c)(functionDefs, annottees : _*)
  }

  private def buildTypeSafeFunction(c: blackbox.Context)
                                (ndarrayfunction: NDArrayFunction): c.universe.DefDef = {
    import c.universe._

    val returnType = "org.apache.mxnet.NDArrayFuncReturn"

    // Construct argument field
    val argDef = ListBuffer[String]()
    argDef ++= buildArgDefs(ndarrayfunction)

    // Construct Implementation field
    var impl = ListBuffer[String]()
    impl += "val map = scala.collection.mutable.Map[String, Any]()"
    impl += "val args = scala.collection.mutable.ArrayBuffer.empty[NDArray]"

    ndarrayfunction.listOfArgs.foreach({ ndarrayarg =>
      // NDArray arg implementation
      val arrayType = "org.apache.mxnet.NDArray"

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
        "map(\"" + ndarrayarg.argName + "\") = " + ndarrayarg.safeArgName
      }
      impl.append(
        if (ndarrayarg.isOptional) s"if (!${ndarrayarg.safeArgName}.isEmpty) $base.get"
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
    var finalStr = s"def ${ndarrayfunction.name}"
    finalStr += s" (${argDef.mkString(",")}) : $returnType"
    finalStr += s" = {${impl.mkString("\n")}}"
    c.parse(finalStr).asInstanceOf[DefDef]
  }

}
