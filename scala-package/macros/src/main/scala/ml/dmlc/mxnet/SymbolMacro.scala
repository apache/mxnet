package ml.dmlc.mxnet

import scala.annotation.StaticAnnotation
import scala.collection.mutable.ListBuffer
import scala.language.experimental.macros
import scala.reflect.macros.blackbox

import ml.dmlc.mxnet.init.Base._

private[mxnet] class AddSymbolFunctions extends StaticAnnotation {
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

    val functionDefs = symbolFunctions map { case (funcName, funcProp) =>
      val functionScope = if (funcName.startsWith("_")) Modifiers(Flag.PRIVATE) else Modifiers()
      // It will generate definition something like,
      // def Concat(name: String = null, attr: Map[String, String] = null)
      //           (args: Symbol*)(kwargs: Map[String, Any] = null)
      DefDef(functionScope, TermName(funcName), List(),
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
    // scalastyle:off println
    println("Atomic Symbol function defination:\n" + docStr)
    // scalastyle:on println
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
