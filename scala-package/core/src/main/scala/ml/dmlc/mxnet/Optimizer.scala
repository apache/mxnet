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

import scala.collection.mutable

object Optimizer {
  def getUpdater(optimizer: Optimizer): MXKVStoreUpdater = {
    new MXKVStoreUpdater with MXKVStoreCachedStates {
      override def update(index: Int, grad: NDArray, weight: NDArray): Unit = {
        val state =
          if (states.contains(index)) {
            states.get(index).get
          } else {
            val newState = optimizer.createState(index, weight)
            states.put(index, newState)
            newState
          }
        optimizer.update(index, weight, grad, state)
      }

      override def dispose(): Unit = {
        states.values.foreach(optimizer.disposeState)
        states.clear()
      }

      override def serializeState(): Array[Byte] = {
        val bos = new ByteArrayOutputStream()
        try {
          val out = new ObjectOutputStream(bos)
          out.writeInt(states.size)
          states.foreach { case (k, v) =>
            if (v != null) {
              out.writeInt(k)
              val stateBytes = optimizer.serializeState(v)
              if (stateBytes == null) {
                out.writeInt(0)
              } else {
                out.writeInt(stateBytes.length)
                out.write(stateBytes)
              }
            }
          }
          out.flush()
          bos.toByteArray
        } finally {
          try {
            bos.close()
          } catch {
            case _: Throwable =>
          }
        }
      }

      override def deserializeState(bytes: Array[Byte]): Unit = {
        val bis = new ByteArrayInputStream(bytes)
        var in: ObjectInputStream = null
        try {
          in = new ObjectInputStream(bis)
          val size = in.readInt()
          (0 until size).foreach(_ => {
            val key = in.readInt()
            val bytesLength = in.readInt()
            val value =
              if (bytesLength > 0) {
                val bytes = Array.fill[Byte](bytesLength)(0)
                in.readFully(bytes)
                optimizer.deserializeState(bytes)
              } else {
                null
              }
            states.update(key, value)
          })
        } finally {
          try {
            if (in != null) {
              in.close()
            }
          } catch {
            case _: Throwable =>
          }
        }
      }
    }
  }
}

abstract class Optimizer extends Serializable {
  protected var lrScale: mutable.Map[Int, Float] = mutable.HashMap.empty[Int, Float]
  protected var numUpdate: Int = 0
  protected val indexUpdateCount: mutable.Map[Int, Int] = mutable.HashMap.empty[Int, Int]

  protected var specialized: Boolean = false
  protected val weightSet: mutable.Set[Int] = mutable.HashSet.empty[Int]
  protected var rescaleGrad: Float = 1
  @transient protected var symbol: Symbol = null
  protected var idx2name: Map[Int, String] = null

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  // TODO: make state a ClassTag
  def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit

  // Create additional optimizer state such as momentum.
  // TODO: make returned state a ClassTag
  def createState(index: Int, weight: NDArray): AnyRef

  // Dispose the state it created
  def disposeState(state: AnyRef): Unit

  def serializeState(state: AnyRef): Array[Byte]

  def deserializeState(bytes: Array[Byte]): AnyRef

  // Set individual learning rate scale for parameters
  def setLrScale(lrScale: Map[Int, Float]) {
    this.lrScale = mutable.Map(lrScale.toSeq: _*)
  }

  def setArgNames(argNames: Seq[String]): Unit = {
    if (argNames != null) {
      specialized = true
      var index = 0
      argNames foreach { name =>
        if (!name.endsWith("data") && !name.endsWith("label")) {
          if (name.endsWith("weight")) {
            weightSet.add(index)
          }
          index += 1
        }
      }
    }
  }

  // Set rescaling factor of gradient.
  def setRescaleGrad(rescaleGrad: Float): Unit = {
    this.rescaleGrad = rescaleGrad
  }

  // TODO
  def setSymbol(sym: Symbol): Unit = {
    this.symbol = sym
  }

  // TODO: Special treat weight decay in parameters.
  def setIdx2Name(paramIdx2Name: Map[Int, String]): Unit = {
    this.idx2name = paramIdx2Name
  }

  /**
   * update num_update
   * @param index The index will be updated
   */
  protected def updateCount(index: Int): Unit = {
    val count = indexUpdateCount.getOrElseUpdate(index, 0) + 1
    indexUpdateCount.update(index, count)
    numUpdate = Math.max(count, numUpdate)
  }

  protected def getWd(index: Int, wd: Float): Float = {
    if (specialized) {
      if (this.weightSet.contains(index)) {
        wd
      } else {
        0f
      }
    } else {
      wd
    }
  }
}

trait MXKVStoreUpdater {
  /**
   * user-defined updater for the kvstore
   * It's this updater's responsibility to delete recv and local
   * @param key the key
   * @param recv the pushed value on this key
   * @param local the value stored on local on this key
   */
  def update(key: Int, recv: NDArray, local: NDArray): Unit
  def dispose(): Unit
  // def serializeState(): Array[Byte]
  // def deserializeState(bytes: Array[Byte]): Unit
}

trait MXKVStoreCachedStates {
  protected val states = new scala.collection.mutable.HashMap[Int, AnyRef]

  /**
   * Serialize states to byte array
   * @return serialized states
   */
  def serializeState(): Array[Byte]

  /**
   * Update states with serialized results
   * @param bytes Generated by serializeState()
   */
  def deserializeState(bytes: Array[Byte]): Unit
}
