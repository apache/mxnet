package ml.dmlc.mxnet

/**
 * Shape of [[NDArray]] or other data
 * @author Yizhi Liu
 */
class Shape(dims: Traversable[Int]) extends Serializable {
  private val shape = dims.toVector

  def this(dims: Int*) = {
    this(dims.toVector)
  }

  def apply(dim: Int): Int = shape(dim)
  def size: Int = shape.size
  def length: Int = shape.length
  def drop(dim: Int): Shape = new Shape(shape.drop(dim))
  def slice(from: Int, end: Int): Shape = new Shape(shape.slice(from, end))
  def product: Int = shape.product
  def head: Int = shape.head

  def ++(other: Shape): Shape = new Shape(shape ++ other.shape)

  def toArray: Array[Int] = shape.toArray
  def toVector: Vector[Int] = shape

  override def toString(): String = s"(${shape.mkString(",")})"

  override def equals(o: Any): Boolean = o match {
    case that: Shape =>
      that != null && that.shape.sameElements(shape)
    case _ => false
  }

  override def hashCode(): Int = {
    shape.hashCode()
  }
}

object Shape {
  def apply(dims: Int *): Shape = new Shape(dims: _*)
  def apply(dims: Traversable[Int]): Shape = new Shape(dims)
}
