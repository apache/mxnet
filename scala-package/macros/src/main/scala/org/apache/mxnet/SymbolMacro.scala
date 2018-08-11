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
import org.apache.mxnet.init.Base._
import org.apache.mxnet.utils.{CToScalaUtils, OperatorBuildUtils}

private[mxnet] class AddSymbolFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolImplMacros.addDefs
}

private[mxnet] class AddSymbolAPIs(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolImplMacros.typeSafeAPIDefs
}

private[mxnet] object SymbolImplMacros {
  case class SymbolArg(argName: String, argType: String, isOptional : Boolean)
  case class SymbolFunction(name: String, listOfArgs: List[SymbolArg])

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(annottees: _*)
  }
  def typeSafeAPIDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    typedAPIImpl(c)(annottees: _*)
  }
  // scalastyle:on havetype

  private val symbolFunctions: List[SymbolFunction] = initSymbolModule()

  /**
    * Implementation for fixed input API structure
    */
  private def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newSymbolFunctions = {
      if (isContrib) symbolFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else symbolFunctions.filter(!_.name.startsWith("_"))
    }


    val functionDefs = newSymbolFunctions map { symbolfunction =>
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

    structGeneration(c)(functionDefs, annottees : _*)
  }

  /**
    * Implementation for Dynamic typed API Symbol.api.<functioname>
    */
  private def typedAPIImpl(c: blackbox.Context)(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolAPIs($b)" => c.eval[Boolean](c.Expr(b))
    }

    // Defines Operators that should not generated
    val notGenerated = Set("Custom")

    // TODO: Put Symbol.api.foo --> Stable APIs
    // Symbol.contrib.bar--> Contrib APIs
    val newSymbolFunctions = {
      if (isContrib) symbolFunctions.filter(
        func => func.name.startsWith("_contrib_") || !func.name.startsWith("_"))
      else symbolFunctions.filter(!_.name.startsWith("_"))
    }.filterNot(ele => notGenerated.contains(ele.name))

    val functionDefs = newSymbolFunctions map { symbolfunction =>

      // Construct argument field
      var argDef = ListBuffer[String]()
      // Construct Implementation field
      var impl = ListBuffer[String]()
      impl += "val map = scala.collection.mutable.Map[String, Any]()"
      impl += "var args = Seq[org.apache.mxnet.Symbol]()"
      symbolfunction.listOfArgs.foreach({ symbolarg =>
        // var is a special word used to define variable in Scala,
        // need to changed to something else in order to make it work
        val currArgName = symbolarg.argName match {
          case "var" => "vari"
          case "type" => "typeOf"
          case default => symbolarg.argName
        }
        if (symbolarg.isOptional) {
          argDef += s"${currArgName} : Option[${symbolarg.argType}] = None"
        }
        else {
          argDef += s"${currArgName} : ${symbolarg.argType}"
        }
        // Symbol arg implementation
        val returnType = "org.apache.mxnet.Symbol"
        val base =
        if (symbolarg.argType.equals(s"Array[$returnType]")) {
          if (symbolarg.isOptional) s"if (!$currArgName.isEmpty) args = $currArgName.get.toSeq"
          else s"args = $currArgName.toSeq"
        } else {
          if (symbolarg.isOptional) {
            // scalastyle:off
            s"if (!$currArgName.isEmpty) map(" + "\"" + symbolarg.argName + "\"" + s") = $currArgName.get"
            // scalastyle:on
          }
          else "map(\"" + symbolarg.argName + "\"" + s") = $currArgName"
        }

        impl += base
      })
      argDef += "name : String = null"
      argDef += "attr : Map[String, String] = null"
      // scalastyle:off
      // TODO: Seq() here allows user to place Symbols rather than normal arguments to run, need to fix if old API deprecated
      impl += "org.apache.mxnet.Symbol.createSymbolGeneral(\"" + symbolfunction.name + "\", name, attr, args, map.toMap)"
      // scalastyle:on
      // Combine and build the function string
      val returnType = "org.apache.mxnet.Symbol"
      var finalStr = s"def ${symbolfunction.name}"
      finalStr += s" (${argDef.mkString(",")}) : $returnType"
      finalStr += s" = {${impl.mkString("\n")}}"
      c.parse(finalStr).asInstanceOf[DefDef]
    }
    structGeneration(c)(functionDefs, annottees : _*)
  }

  /**
    * Generate class structure for all function APIs
    * @param c
    * @param funcDef DefDef type of function definitions
    * @param annottees
    * @return
    */
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
  private def initSymbolModule(): List[SymbolFunction] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    // TODO: Add '_linalg_', '_sparse_', '_image_' support
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName)
    }).toList
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle, aliasName: String)
      : SymbolFunction = {
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
      println("Symbol function definition:\n" + docStr)
    }
    // scalastyle:on println
    val argList = argNames zip argTypes map { case (argName, argType) =>
        val typeAndOption =
          CToScalaUtils.argumentCleaner(argName, argType, "org.apache.mxnet.Symbol")
        new SymbolArg(argName, typeAndOption._1, typeAndOption._2)
    }
    new SymbolFunction(aliasName, argList.toList)
  }
}
