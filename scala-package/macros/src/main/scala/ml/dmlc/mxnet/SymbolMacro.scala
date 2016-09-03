package ml.dmlc.mxnet

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context


import ml.dmlc.mxnet.init.Base._

private[mxnet] class FillDefs extends StaticAnnotation {
  def macroTransform(annottees: Any*) = macro ImplMacros.addDefs
}

object ImplMacros {
  case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)


  def addDefs(c: Context)(annottees: c.Expr[Any]*) = {
    impl(c)(false, annottees: _*)
  }

  /*
  def LeakyReLU(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LeakyReLU", name, attr)
  }
  */
  val symbolFunctions: Map[String, SymbolFunction] = initSymbolModule()

  def impl(c: Context)(addSuper: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    val AST_TYPE_MAP_STRING_ANY = AppliedTypeTree(Ident(TypeName("Map")),
      List(Ident(TypeName("String")), Ident(TypeName("Any"))))
    val AST_TYPE_MAP_STRING_STRING = AppliedTypeTree(Ident(TypeName("Map")),
      List(Ident(TypeName("String")), Ident(TypeName("String"))))

    val inputs = annottees.map(_.tree).toList
    // create the definitions we're going to add
    val newDefDefs = List(
      DefDef(Modifiers(), TermName("x"), List(), List(), TypeTree(), Literal(Constant(5))),
      DefDef(Modifiers(), TermName("y"), List(), List(), TypeTree(), Literal(Constant(7.0f))),
      DefDef(Modifiers(), TermName("f"), List(), List(List(ValDef(Modifiers(),
        TermName("a"), Ident(TypeName("Int")), EmptyTree))), TypeTree(),
        Apply(Select(Ident(TermName("a")), TermName("$plus")), List(Literal(Constant(3))))),
      DefDef(Modifiers(), TermName("f2"), List(), List(List(ValDef(Modifiers(),
        TermName("a"), Ident(TypeName("Int")), EmptyTree))), TypeTree(),
        Apply(Select(Ident(TermName("a")), TermName("$plus")), List(Ident(TermName("b"))))),
      DefDef(Modifiers(), TermName("f3"), List(), List(List(ValDef(Modifiers(),
        TermName("a"), Ident(TypeName("Int")), EmptyTree))), TypeTree(),
        Apply(Ident(TermName("showA")), List(Ident(TermName("a"))))),
      DefDef(Modifiers(), TermName("f4"), List(), List(List(ValDef(Modifiers(),
        TermName("a"), Ident(TypeName("Int")), EmptyTree),
        ValDef(Modifiers(), TermName("b"), Ident(TypeName("String")), EmptyTree))), TypeTree(),
        Apply(Select(Ident(TermName("a")), TermName("$plus")), List(Ident(TermName("b"))))),
      DefDef(Modifiers(), TermName("LeakyReLU2"), List(),
        List(
          List(
            ValDef(Modifiers(Flag.PARAM | Flag.DEFAULTPARAM), TermName("name"), Ident(TypeName("String")), Literal(Constant(null))),
            ValDef(Modifiers(Flag.PARAM | Flag.DEFAULTPARAM), TermName("attr"), AST_TYPE_MAP_STRING_STRING, Literal(Constant(null)))
          ),
          List(
            ValDef(Modifiers(), TermName("kwargs"), AST_TYPE_MAP_STRING_ANY, EmptyTree)
          )
        ), TypeTree(),
        Apply(
          Apply(
            Ident(TermName("createFromNamedSymbolsNoCheck")),
            List(Literal(Constant("LeakyReLU")))
          ),
          List(Ident(TermName("kwargs")))
        )
      )
    )

    // pattern match on the inputs
    val modDefs = inputs map { tree => tree match {
      case ClassDef(mods, name, something, template) =>
        // println(s"DEBUG: $mods | $name | $something | $template")
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            Template(superMaybe, emptyValDef, defs ++ newDefDefs)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ClassDef(mods, name, something, q)
      case ModuleDef(mods, name, template) =>
        // println(s"DEBUG Module: $mods | $name | $template")
        val q = template match {
          case Template(superMaybe, emptyValDef, defs) =>
            // println(s"DEBUG Template: $superMaybe | $emptyValDef | $defs")
            Template(superMaybe, emptyValDef, defs ++ newDefDefs)
          case ex =>
            throw new IllegalArgumentException(s"Invalid template: $ex")
        }
        ModuleDef(mods, name, q)
      case ex =>
        throw new IllegalArgumentException(s"Invalid macro input: $ex")
    }
    }
    // wrap the result up in an Expr, and return it
    val result = c.Expr(Block(modDefs, Literal(Constant())))
    result
  }

  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): Map[String, SymbolFunction] = {
    val symbolList = ListBuffer.empty[SymbolHandle]
    _LIB.mxSymbolListAtomicSymbolCreators(symbolList)
    symbolList.map(makeAtomicSymbolFunction).toMap
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle): (String, SymbolFunction) = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    _LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs)
    val paramStr = ctypes2docstring(argNames, argTypes, argDescs)
    val extraDoc: String = if (keyVarNumArgs.value != null && keyVarNumArgs.value.length > 0) {
        s"This function support variable length of positional input (${keyVarNumArgs.value})."
      } else {
        ""
      }
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n$extraDoc\n"
    println("Atomic Symbol function defination:\n" + docStr)
    (name.value, new SymbolFunction(handle, keyVarNumArgs.value))
  }

  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(argNames: Seq[String],
                       argTypes: Seq[String],
                       argDescs: Seq[String]): String = {
    val params =
      (argNames zip argTypes zip argDescs) map { case ((argName, argType), argDesc) =>
        val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"$argName : $argType$desc"
      }
    s"Parameters\n----------\n${params.mkString("\n")}\n"
  }
}
