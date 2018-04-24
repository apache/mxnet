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
import org.apache.mxnet.utils.OperatorBuildUtils

private[mxnet] class AddSymbolFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolImplMacros.addDefs
}

private[mxnet] object SymbolImplMacros {
  case class SymbolArg(argName: String, argType: String, isOptional : Boolean)
  case class SymbolFunction(name: String, listOfArgs: List[SymbolArg])

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(false, annottees: _*)
  }
  // scalastyle:off havetype

  private val symbolFunctions: List[SymbolFunction] = initSymbolModule()

  private def impl(c: blackbox.Context)(addSuper: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newSymbolFunctions = {
      if (isContrib) symbolFunctions.filter(_.name.startsWith("_contrib_"))
      else symbolFunctions.filter(!_.name.startsWith("_contrib_"))
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
         """
    }

    val newFunctionDefs = newSymbolFunctions map { symbolfunction =>
      // TODO: Implement the codeGen
      null
    }


    val inputs = annottees.map(_.tree).toList
    // pattern match on the inputs
    val modDefs = inputs map {
      case ClassDef(mods, name, something, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ functionDefs)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ functionDefs)
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

  // Convert C++ Types to Scala Types
  private def typeConversion(in : String) : String = {
    in match {
      case "Shape(tuple)" | "ShapeorNone" => "Shape"
      case "Symbol" | "NDArray" | "NDArray-or-Symbol" => "Symbol"
      case "Symbol[]" | "NDArray[]" | "NDArray-or-Symbol[]" | "SymbolorSymbol[]" => "Array[Symbol]"
      case "float" | "real_t" => "MXFloat"
      case "int" | "intorNone" | "int(non-negative)" => "Int"
      case "long" | "long(non-negative)" => "Long"
      case "double" => "Double"
      case "string" => "String"
      case "boolean" => "Boolean"
      case "tupleof<float>" => "Any"
      case default => throw new IllegalArgumentException(s"Invalid type for args: $default")
    }
  }

  private def argumentCleaner(argType : String) : (String, Boolean) = {
    val spaceRemoved = argType.replaceAll("\\s+", "")
    var commaRemoved : Array[String] = new Array[String](0)
    // Deal with the case e.g: stype : {'csr', 'default', 'row_sparse'}
    if (spaceRemoved.charAt(0)== '{') {
      val endIdx = spaceRemoved.indexOf('}')
      commaRemoved = spaceRemoved.substring(endIdx + 1).split(",")
      // commaRemoved(0) = spaceRemoved.substring(0, endIdx+1)
      commaRemoved(0) = "string"
    } else {
      commaRemoved = spaceRemoved.split(",")
    }
    // Optional Field
    if (commaRemoved.length == 3) {
      (typeConversion(commaRemoved(0)), true)
      // TODO: Qing: do we set default value on our side?
      // optionalField = " = " + conversion(typeConv, commaRemoved(2).split("=")(1))
    } else if (commaRemoved.length == 2) {
      val tempType = typeConversion(argType)
      val tempOptional = tempType.equals("Symbol")
      (commaRemoved(0), tempOptional)
    } else {
      throw new IllegalArgumentException(s"Unrecognized arg field: $argType")
    }

  }


  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): List[SymbolFunction] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
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
    val argList = (argNames zip argTypes) map { case ((argName, argType)) =>
        val tup = argumentCleaner(argType)
        new SymbolArg(argName, tup._1, tup._2)
    }
    new SymbolFunction(aliasName, argList.toList)
  }
}
