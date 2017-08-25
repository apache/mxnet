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

import java.io._
import java.nio.ByteBuffer
import java.nio.charset.Charset

import org.apache.commons.codec.binary.Base64

import scala.reflect.ClassTag

/**
 * Serialize & deserialize Java/Scala [[Serializable]] objects
 */
private[mxnet] abstract class Serializer {
  def serialize[T: ClassTag](t: T): ByteBuffer
  def deserialize[T: ClassTag](bytes: ByteBuffer): T
}

private[mxnet] object Serializer {
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

private[mxnet] class JavaSerializer extends Serializer {
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
