package ml.dmlc.mxnet

import java.io._
import java.nio.ByteBuffer
import java.nio.charset.Charset

import org.apache.commons.codec.binary.Base64

import scala.reflect.ClassTag

/**
 * Serialize & deserialize Java/Scala [[Serializable]] objects
 * @author Yizhi Liu
 */
abstract class Serializer {
  def serialize[T: ClassTag](t: T): ByteBuffer
  def deserialize[T: ClassTag](bytes: ByteBuffer): T
}

object Serializer {
  val UTF8 = Charset.forName("UTF-8")

  def getSerializer: Serializer = getSerializer(None)

  def getSerializer(serializer: Serializer): Serializer = {
    // TODO: dynamically get from mxnet env to support other serializers like Kyro
    if (serializer == null) new JavaSerializer else serializer
  }

  def getSerializer(serializer: Option[Serializer]): Serializer = {
    // TODO: dynamically get from mxnet env to support other serializers like Kyro
    serializer.getOrElse(new JavaSerializer)
  }

  def encodeBase64String(bytes: ByteBuffer): String = {
    new String(Base64.encodeBase64(bytes.array), UTF8)
  }

  def decodeBase64String(str: String): ByteBuffer = {
    ByteBuffer.wrap(Base64.decodeBase64(str.getBytes(UTF8)))
  }
}

class JavaSerializer extends Serializer {
  override def serialize[T: ClassTag](t: T): ByteBuffer = {
    val bos = new ByteArrayOutputStream()
    val out = new ObjectOutputStream(bos)
    out.writeObject(t)
    out.close()
    ByteBuffer.wrap(bos.toByteArray)
  }

  override def deserialize[T: ClassTag](bytes: ByteBuffer): T = {
    val byteArray = bytes.array()
    val bis = new ByteArrayInputStream(byteArray)
    val in = new ObjectInputStream(bis)
    in.readObject().asInstanceOf[T]
  }
}
