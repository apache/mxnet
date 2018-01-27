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
import scala.util.Either

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
  protected val lrMult: mutable.Map[Either[Int, String], Float] =
    mutable.HashMap.empty[Either[Int, String], Float]
  protected val wdMult: mutable.Map[Either[Int, String], Float] =
    mutable.HashMap.empty[Either[Int, String], Float]
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
  @deprecated("Use setLrMult instead.")
  def setLrScale(lrScale: Map[Int, Float]): Unit = {
    val argsLrScale: Map[Either[Int, String], Float] = lrScale.map { case (k, v) => Left(k) -> v }
    setLrMult(argsLrScale)
  }

  /**
   * Sets an individual learning rate multiplier for each parameter.
   * If you specify a learning rate multiplier for a parameter, then
   * the learning rate for the parameter will be set as the product of
   * the global learning rate and its multiplier.
   * note:: The default learning rate multiplier of a `Variable`
   * can be set with `lr_mult` argument in the constructor.
   * @param argsLrMult: Map[Either[Int, String], Float]
   *                  For each of its key-value entries, the learning rate multipler for the
   *                  parameter specified in the key will be set as the given value.
   *
   *                  You can specify the parameter with either its name or its index.
   *                  If you use the name, you should also call the `setSymbol` method first,
   *                  and the name you specified in the key of `argsLrMult` should match
   *                  the name of the parameter in the `sym` you pass to `setSymbol` method.
   *                  If you use the index, it should correspond to the index of the parameter
   *                  used in the `update` method.
   *
   *                  Specifying a parameter by its index is only supported for backward
   *                  compatibility, and we recommend to use the name instead.
   */
  def setLrMult(argsLrMult: Map[Either[Int, String], Float]): Unit = {
    argsLrMult.foreach { case (k, v) => this.lrMult(k) = v }
  }

  /**
   * Sets an individual weight decay multiplier for each parameter.
   *
   * By default, the weight decay multipler is set as 0 for all
   * parameters whose name don't end with ``_weight`` or ``_gamma``, if
   * you call the `setIdx2Name` method to set idx2name.
   *
   * note:: The default weight decay multiplier for a `Variable`
   * can be set with its `wd_mult` argument in the constructor.
   * @param argsWdMult: Map[Either[Int, String], Float]
   *                  For each of its key-value entries, the learning rate multipler for the
   *                  parameter specified in the key will be set as the given value.
   *
   *                  You can specify the parameter with either its name or its index.
   *                  If you use the name, you should also call the `setSymbol` method first,
   *                  and the name you specified in the key of `argsWdMult` should match
   *                  the name of the parameter in the `sym` you pass to `setSymbol` method.
   *                  If you use the index, it should correspond to the index of the parameter
   *                  used in the `update` method.
   *
   *                  Specifying a parameter by its index is only supported for backward
   *                  compatibility, and we recommend to use the name instead.
   */
  def setWdMult(argsWdMult: Map[Either[Int, String], Float]): Unit = {
    argsWdMult.foreach { case (k, v) => this.wdMult(k) = v }
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

  def setSymbol(sym: Symbol): Unit = {
    this.symbol = sym
    if (this.symbol != null) {
      val attr = this.symbol.attrMap
      for (name <- this.symbol.listArguments()) {
        if (attr.contains(name) && attr(name).contains("__lr_mult__")) {
          this.lrMult(Right(name)) = attr(name)("__lr_mult__").toFloat
        }
        if (attr.contains(name) && attr(name).contains("__wd_mult__")) {
          this.wdMult(Right(name)) = attr(name)("__wd_mult__").toFloat
        }
      }
    }
  }

  def setIdx2Name(paramIdx2Name: Map[Int, String]): Unit = {
    this.idx2name = paramIdx2Name
    if (this.idx2name != null) {
      for (n <- this.idx2name.values) {
        if (!(n.endsWith("_weight") || n.endsWith("_gamma"))) {
          this.wdMult(Right(n)) = 0f
        }
      }
    }
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

 // Gets the learning rate given the index of the weight.
  protected def getLr(index: Int, lr: Float): Float = {
    var llr = lr
    if (this.lrMult.contains(Left(index))) {
      llr *= this.lrMult(Left(index))
    } else if (this.idx2name != null && this.idx2name.contains(index)) {
      llr *= this.lrMult.getOrElse(Right(this.idx2name(index)), 1.0f)
    }
    llr
  }

  // Gets weight decay for index.
  protected def getWd(index: Int, wd: Float): Float = {
    var lwd = if (specialized) {
      if (this.weightSet.contains(index)) {
        wd
      } else {
        0f
      }
    } else {
      wd
    }
    if (this.wdMult.contains(Left(index))) {
      lwd *= this.wdMult(Left(index))
    } else if (this.idx2name != null && this.idx2name.contains(index)) {
      lwd *= this.wdMult.getOrElse(Right(this.idx2name(index)), 1.0f)
    }
    lwd
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
