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

import org.apache.mxnet.GeneratorBase

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

private[mxnet] class AddJNDArrayAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro JavaNDArrayMacro.typeSafeAPIDefs
}

private[mxnet] object JavaNDArrayMacro extends GeneratorBase {

  // scalastyle:off havetype
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typeSafeAPIImpl(c)(annottees: _*)
  }
  // scalastyle:off havetype

  private def typeSafeAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddJNDArrayAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }
    // Defines Operators that should not generated
    val notGenerated = Set("Custom")

    val newNDArrayFunctions = functionsToGenerate(false, false, true)
      .filterNot(ele => notGenerated.contains(ele.name)).groupBy(_.name.toLowerCase).map(ele => {
      /* Pattern matching for not generating deprecated method
       * Group all method name in lowercase
       * Kill the capital lettered method such as Cast vs cast
       * As it defined by default it deprecated
       */
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
      // Construct argument field with all required args
      var argDef = ListBuffer[String]()
      // Construct function Implementation field (e.g norm)
      var impl = ListBuffer[String]()
      impl += "val map = scala.collection.mutable.Map[String, Any]()"
      impl +=
        "val args= scala.collection.mutable.ArrayBuffer.empty[org.apache.mxnet.NDArray]"
      ndarrayfunction.listOfArgs.foreach({ ndarrayArg =>
        // var is a special word used to define variable in Scala,
        // need to changed to something else in order to make it work
        var currArgName = ndarrayArg.safeArgName
        if (useParamObject) currArgName = s"po.get${currArgName.capitalize}()"
        argDef += s"$currArgName : ${ndarrayArg.argType}"
        // NDArray arg implementation
        val returnType = "org.apache.mxnet.javaapi.NDArray"
        val base =
          if (ndarrayArg.argType.equals(returnType)) {
            s"args += $currArgName"
          } else if (ndarrayArg.argType.equals(s"Array[$returnType]")){
            s"$currArgName.foreach(args+=_)"
          } else {
            "map(\"" + ndarrayArg.argName + "\") = " + currArgName
          }
        impl.append(
          if (ndarrayArg.isOptional) s"if ($currArgName != null) $base"
          else base
        )
      })
      // add default out parameter
      argDef += s"out: org.apache.mxnet.javaapi.NDArray"
      if (useParamObject) {
        impl += "if (po.getOut() != null) map(\"out\") = po.getOut().nd"
      } else {
        impl += "if (out != null) map(\"out\") = out.nd"
      }
      val returnType = "Array[org.apache.mxnet.javaapi.NDArray]"
      // scalastyle:off
      // Combine and build the function string
      impl += "val finalArr = org.apache.mxnet.NDArray.genericNDArrayFunctionInvoke(\"" +
        ndarrayfunction.name + "\", args.toSeq, map.toMap).arr"
      impl += "finalArr.map(ele => new NDArray(ele))"
      if (useParamObject) {
        val funcDef =
          s"""def ${ndarrayfunction.name}(po: ${ndarrayfunction.name}Param): $returnType = {
             |  ${impl.mkString("\n")}
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
    structGeneration(c)(functionDefs.toList, annottees : _*)
  }
}
