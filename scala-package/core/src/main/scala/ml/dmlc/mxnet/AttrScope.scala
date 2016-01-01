package ml.dmlc.mxnet

object AttrScope {
  private var _current = new AttrScope()
  def current: AttrScope = _current
  private def setCurrentAttr(attr: AttrScope): Unit = {
    _current = attr
  }
}

/**
 * Attribute manager for scoping.
 * User can also inherit this object to change naming behavior.
 * @author Yizhi Liu
 */
class AttrScope(val attr: Map[String, String] = Map.empty) {
  private var _attr = attr
  /**
   * Get the attribute dict given the attribute set by the symbol.
   * @param userDefinedAttr The attribute passed in by user during symbol creation.
   * @return Updated attributes to add other scope related attributes.
   */
  def get(userDefinedAttr: Map[String, String]): Map[String, String] = {
    attr ++ userDefinedAttr
  }

  def withScope[T](body: => T): T = {
    val oldAttrScope = AttrScope.current
    this._attr = AttrScope.current.attr ++ this._attr
    AttrScope.setCurrentAttr(this)
    try {
      body
    } finally {
      AttrScope.setCurrentAttr(oldAttrScope)
    }
  }
}
