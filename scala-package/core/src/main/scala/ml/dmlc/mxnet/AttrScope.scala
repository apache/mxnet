package ml.dmlc.mxnet

object AttrScope {
  private var _current = new AttrScope()
  def current: AttrScope = _current
  private def setCurrentAttr(attr: AttrScope): Unit = {
    _current = attr
  }

  def withScope[T](attr: Map[String, String])(body: => T): T = {
    val oldAttrScope = AttrScope.current
    val updatedAttr = AttrScope.current.attr ++ attr
    AttrScope.setCurrentAttr(new AttrScope(updatedAttr))
    val ret = body
    AttrScope.setCurrentAttr(oldAttrScope)
    ret
  }
}

/**
 * Attribute manager for scoping.
 * User can also inherit this object to change naming behavior.
 * @author Yizhi Liu
 */
class AttrScope(private var attr: Map[String, String] = Map.empty) {
  /**
   * Get the attribute dict given the attribute set by the symbol.
   * @param userDefinedAttr The attribute passed in by user during symbol creation.
   * @return Updated attributes to add other scope related attributes.
   */
  def get(userDefinedAttr: Map[String, String]): Map[String, String] = {
    attr ++ userDefinedAttr
  }
}
