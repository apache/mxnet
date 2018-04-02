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

package ml.dmlc.mxnet

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

import ml.dmlc.mxnet.init.Base._
import ml.dmlc.mxnet.utils.OperatorBuildUtils

private[mxnet] class AddSymbolFunctions(isContrib: Boolean) extends StaticAnnotation {
  private[mxnet] def macroTransform(annottees: Any*) = macro SymbolImplMacros.addDefs
}

private[mxnet] object SymbolImplMacros {
  case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)

  // scalastyle:off havetype
  def addDefs(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    impl(c)(false, annottees: _*)
  }
  // scalastyle:off havetype

  private val symbolFunctions: Map[String, SymbolFunction] = initSymbolModule()

  private def impl(c: blackbox.Context)(addSuper: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val isContrib: Boolean = c.prefix.tree match {
      case q"new AddSymbolFunctions($b)" => c.eval[Boolean](c.Expr(b))
    }

    val newSymbolFunctions = {
      if (isContrib) symbolFunctions.filter(_._1.startsWith("_contrib_"))
      else symbolFunctions.filter(!_._1.startsWith("_contrib_"))
    }

    val AST_TYPE_MAP_STRING_ANY = AppliedTypeTree(Ident(TypeName("Map")),
      List(Ident(TypeName("String")), Ident(TypeName("Any"))))
    val AST_TYPE_MAP_STRING_STRING = AppliedTypeTree(Ident(TypeName("Map")),
      List(Ident(TypeName("String")), Ident(TypeName("String"))))
    val AST_TYPE_SYMBOL_VARARG = AppliedTypeTree(
      Select(
        Select(Ident(termNames.ROOTPKG), TermName("scala")),
        TypeName("<repeated>")
      ),
      List(Select(Select(Select(
        Ident(TermName("ml")), TermName("dmlc")), TermName("mxnet")), TypeName("Symbol")))
    )

    val functionDefs = newSymbolFunctions map { case (funcName, funcProp) =>
      val functionScope = {
        if (isContrib) Modifiers()
        else {
          if (funcName.startsWith("_")) Modifiers(Flag.PRIVATE) else Modifiers()
        }
      }
      val newName = {
        if (isContrib) funcName.substring(funcName.indexOf("_contrib_") + "_contrib_".length())
        else funcName
      }

      // It will generate definition something like,
      // def Concat(name: String = null, attr: Map[String, String] = null)
      //           (args: Symbol*)(kwargs: Map[String, Any] = null)
      DefDef(functionScope, TermName(newName), List(),
        List(
          List(
            ValDef(Modifiers(Flag.PARAM | Flag.DEFAULTPARAM), TermName("name"),
              Ident(TypeName("String")), Literal(Constant(null))),
            ValDef(Modifiers(Flag.PARAM | Flag.DEFAULTPARAM), TermName("attr"),
              AST_TYPE_MAP_STRING_STRING, Literal(Constant(null)))
          ),
          List(
            ValDef(Modifiers(), TermName("args"), AST_TYPE_SYMBOL_VARARG, EmptyTree)
          ),
          List(
            ValDef(Modifiers(Flag.PARAM | Flag.DEFAULTPARAM), TermName("kwargs"),
              AST_TYPE_MAP_STRING_ANY, Literal(Constant(null)))
          )
        ), TypeTree(),
        Apply(
          Ident(TermName("createSymbolGeneral")),
          List(
            Literal(Constant(funcName)),
            Ident(TermName("name")),
            Ident(TermName("attr")),
            Ident(TermName("args")),
            Ident(TermName("kwargs"))
          )
        )
      )
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

  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): Map[String, SymbolFunction] = {
    val opNames = ListBuffer.empty[String]
    _LIB.mxListAllOpNames(opNames)
    opNames.map(opName => {
      val opHandle = new RefLong
      _LIB.nnGetOpHandle(opName, opHandle)
      makeAtomicSymbolFunction(opHandle.value, opName)
    }).toMap
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle, aliasName: String)
      : (String, SymbolFunction) = {
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
    (aliasName, new SymbolFunction(handle, keyVarNumArgs.value))
  }
}
