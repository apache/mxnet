package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import java.io.File
import scala.io.Source
import java.io.PrintWriter
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.DataInputStream
import java.io.ByteArrayInputStream

/**
 * Scala interface for read/write RecordIO data format
 *
 * @author Depeng Liang
 *
 * @param uri, path to recordIO file.
 * @param flag, RecordIO.IORead for reading or RecordIO.Write for writing.
 */
class MXRecordIO(uri: String, flag: MXRecordIO.IOFlag) {
  protected val recordIOHandle: RecordIOHandleRef = new RecordIOHandleRef
  protected var isOpen: Boolean = false

  open()

  // Open record file
  protected def open(): Unit = {
    flag match {
      case MXRecordIO.IOWrite => {
        checkCall(_LIB.mxRecordIOWriterCreate(uri, recordIOHandle))
      }
      case MXRecordIO.IORead => {
        checkCall(_LIB.mxRecordIOReaderCreate(uri, recordIOHandle))
      }
    }
    this.isOpen = true
  }

  // Close record file
  def close(): Unit = {
    if (this.isOpen) {
      flag match {
        case MXRecordIO.IOWrite => {
          checkCall(_LIB.mxRecordIOWriterFree(recordIOHandle.value))
        }
        case MXRecordIO.IORead => {
          checkCall(_LIB.mxRecordIOReaderFree(recordIOHandle.value))
        }
      }
    }
  }

  // Reset pointer to first item.
  // If record is opened with RecordIO.IOWrite, this will truncate the file to empty.
  def reset(): Unit = {
    this.close()
    this.open()
  }

  // Write a string buffer as a record
  def write(buf: String): Unit = {
    assert(this.flag == MXRecordIO.IOWrite)
    checkCall(_LIB.mxRecordIOWriterWriteRecord(this.recordIOHandle.value, buf, buf.size))
  }

  // Read a record as string
  def read(): String = {
    assert(this.flag == MXRecordIO.IORead)
    val result = new RefString
    checkCall(_LIB.mxRecordIOReaderReadRecord(this.recordIOHandle.value, result))
    result.value
  }
}

object MXRecordIO {
  sealed trait IOFlag
  case object IOWrite extends IOFlag
  case object IORead extends IOFlag

  case class IRHeader(flag: Int, label: Array[Float], id: Int, id2: Int)

  /**
   * pack an string into MXImageRecord.
   * @param
   *  header of the image record.
   *  header.label an array.
   * @param s string to pack
   * @return the resulting packed string
   */
  def pack(header: IRHeader, s: String): String = {
    val data = new ByteArrayOutputStream()
    val stream = new DataOutputStream(data)
    stream.writeInt(header.label.length)
    header.label.foreach(stream.writeFloat)
    stream.writeInt(header.id)
    stream.writeInt(header.id2)
    stream.writeUTF(s)
    stream.flush()
    stream.close()
    data.toByteArray().map(_.toChar).mkString
  }

  /**
   * unpack a MXImageRecord to string.
   * @param s string buffer from MXRecordIO.read
   * @return
   * header : IRHeader, header of the image record
   * str : String, unpacked string
   */
  def unpack(s: String): (IRHeader, String) = {
    val data = s.toCharArray().map(_.toByte)
    val stream = new DataInputStream(new ByteArrayInputStream(data))
    val flag = stream.readInt()
    val label = (0 until flag).map( idx => stream.readFloat()).toArray
    val id = stream.readInt()
    val id2 = stream.readInt()
    val str = stream.readUTF()
    stream.close()
    (IRHeader(flag, label, id, id2), str)
  }

}

/**
 * Scala interface for read/write RecordIO data formmat with index.
 * Support random access.
 *
 * @author Depeng Liang
 *
 * @param idx_path, path to index file
 * @param uri, path to recordIO file.
 * @param flag, RecordIO.IORead for reading or RecordIO.Write for writing.
 * @param keyType, data type for keys.
 */
class MXIndexedRecordIO(idxPath: String, uri: String, flag: MXRecordIO.IOFlag,
  keyType: MXIndexedRecordIO.KeyType = MXIndexedRecordIO.TyepInt) extends MXRecordIO(uri, flag) {
  private var idx = this.keyType match {
    case MXIndexedRecordIO.TyepInt => Map[Int, Int]()
    case _ => Map[Any, Int]()
  }

  if (flag == MXRecordIO.IORead && new File(idxPath).isFile()) {
    Source.fromFile(idxPath).getLines().foreach { line =>
      val (k, v) = {
        val tmp = line.trim().split("\t")
        val key = this.keyType match {
          case MXIndexedRecordIO.TyepInt => tmp(0).toInt
        }
        (key, tmp(1).toInt)
      }
      this.idx = this.idx + (k -> v)
    }
  }

  override def close(): Unit = {
    if (this.flag == MXRecordIO.IOWrite) {
      val fOut = new PrintWriter(idxPath)
      this.idx.foreach { case (k, v) =>
        fOut.write(s"$k\t$v\n")
      }
      fOut.flush()
      fOut.close()
    }
    super.close()
  }

  override def reset(): Unit = {
    this.idx = Map[Any, Int]()
    super.close()
    super.open()
  }

  // Query current read head position
  def seek(idx: Any): Unit = {
    assert(this.flag == MXRecordIO.IORead)
    val idxx = this.keyType match {
      case MXIndexedRecordIO.TyepInt => idx.asInstanceOf[Int]
    }
    val pos = this.idx(idxx)
    checkCall(_LIB.mxRecordIOReaderSeek(this.recordIOHandle.value, pos))
  }

  // Query current write head position
  def tell(): Int = {
    assert(this.flag == MXRecordIO.IOWrite)
    val pos = new RefInt
    checkCall(_LIB.mxRecordIOWriterTell(this.recordIOHandle.value, pos))
    pos.value
  }

  // Read record with index
  def readIdx(idx: Any): String = {
    this.seek(idx)
    this.read()
  }

  // Write record with index
  def writeIdx(idx: Any, buf: String): Unit = {
    val pos = this.tell()
    val idxx = this.keyType match {
      case MXIndexedRecordIO.TyepInt => idx.asInstanceOf[Int]
    }
    this.idx = this.idx + (idxx -> pos)
    this.write(buf)
  }

  // List all keys from index
  def keys(): Iterable[Any] = this.idx.keys
}

object MXIndexedRecordIO {
  sealed trait KeyType
  case object TyepInt extends KeyType
}
