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
  def addDefs(c: Context)(annottees: c.Expr[Any]*) = {
    impl(c)(false, annottees: _*)
  }

  /*
  def LeakyReLU(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LeakyReLU", name, attr)
  }
  */
  initSymbolModule().foreach(addr => println(s"Symbol addr: $addr"))

  def impl(c: Context)(addSuper: Boolean, annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
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
      DefDef(Modifiers(), TermName("LeakyReLU2"), List(), List(List(
        ValDef(Modifiers(), TermName("name"), Ident(TypeName("String")), EmptyTree),
        ValDef(Modifiers(), TermName("kwargs"),
          AppliedTypeTree(Ident(TypeName("Map")),
            List(Ident(TypeName("String")), Ident(TypeName("Any")))), EmptyTree))), TypeTree(),
        Apply(
          Apply(Ident(TermName("createFromNamedSymbolsNoCheck")),
          List(Literal(Constant("LeakyReLU")))),
        List(Ident(TermName("kwargs")))
        ))
      /*
      DefDef(Modifiers(), TermName("f"), List(), List(List(ValDef(Modifiers(),
        TermName("a"), Ident(TypeName("Int")), EmptyTree)),
        List(ValDef(Modifiers(), TermName("b"),
        Ident(TypeName("String")), EmptyTree))), TypeTree(),
        Apply(Select(Ident(TermName("a")), TermName("$plus")), List(Ident(TermName("b")))))
      DefDef(Modifiers(), newTermName("f"), List(),
        List(List(ValDef(Modifiers(PARAM), newTermName("a"),
        Ident(newTypeName("Int")), EmptyTree)), List(ValDef(Modifiers(PARAM),
        newTermName("b"), Ident(newTypeName("String")), EmptyTree))),
        TypeTree(), Apply(Select(Ident(newTermName("a")), newTermName("$plus")),
        List(Ident(newTermName("b")))))
      */
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

  private def initSymbolModule(): List[Long] = {
    val symbolList = ListBuffer.empty[Long]
    _LIB.mxSymbolListAtomicSymbolCreators(symbolList)
    symbolList.toList
  }
}
