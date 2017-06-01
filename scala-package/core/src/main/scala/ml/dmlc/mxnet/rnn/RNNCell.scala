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

package ml.dmlc.mxnet.rnn

import ml.dmlc.mxnet.{NDArray, Shape, Symbol}

import scala.collection.mutable

/**
 * Container for holding variables.
 * Used by RNN cells for parameter sharing between cells.
 * @param prefix All variables' name created by this container will be prepended with prefix
 */
class RNNParams(private val prefix: String = "") {
  private val params = mutable.HashMap.empty[String, Symbol]
  /**
   * Get a variable with name or create a new one if missing.
   * @param name name of the variable
   * @param attr more arguments that's passed to symbol.Variable
   */
  def get(name: String, attr: Map[String, String] = null): Symbol = {
    val fullName = prefix + name
    val existVar = params.get(fullName)
    existVar.getOrElse {
      val newVar = Symbol.Variable(fullName, attr)
      params.put(name, newVar)
      newVar
    }
  }
}

/**
 * Abstract base class for RNN cells
 * @param prefix prefix for name of layers (and name of weight if params is None)
 * @param inputParams container for weight sharing between cells. created if None.
 */
abstract class BaseRNNCell(protected val prefix: String = "",
                           inputParams: Option[RNNParams],
                           protected val numHidden: Int) {
  private var (myParams, ownParams) =
    inputParams.map(p => (p, false)).getOrElse((new RNNParams(prefix), true))
  private var modified = false
  private var initCounter = -1
  protected var counter = -1
  reset()

  private def normalizeSequence(inputs: Symbol*)(length: Int, layout: String, merge: Boolean,
                                inLayout: Option[String] = None): (IndexedSeq[Symbol], Int) = {
    require(inputs != null, "unroll(inputs=null) has been deprecated." +
      "Please create input variables outside unroll.")
    val axis = layout.indexOf("T")
    val inAxis = inLayout.map(_.indexOf("T")).getOrElse(axis)
    if (inputs.length == 1) {
      val input = inputs(0)
      if (!merge) {
        require(input.listOutputs().length == 1, "unroll doesn't allow grouped symbol as input." +
          "Please convert to list with list(inputs) first or let unroll handle splitting.")
        val splits: Symbol = Symbol.split()(input)(
          Map("axis" -> inAxis, "num_outputs" -> length, "squeeze_axis" -> 1))
        ((0 until length).map(i => splits.get(i)), axis)
      } else {
        if (axis != inAxis) {
          val ret: Symbol = Symbol.swapaxes()(input)(Map("dim0" -> axis, "dim1" -> inAxis))
          (IndexedSeq(ret), axis)
        } else {
          (inputs.toIndexedSeq, axis)
        }
      }
    } else {
      require(length < 0 || length == inputs.length)
      if (merge) {
        val expandedInputs: Seq[Symbol]
          = inputs.map(input => Symbol.expand_dims()(input)(Map("axis" -> axis)))
        val res: Symbol = Symbol.Concat()(expandedInputs: _*)(Map("dim" -> axis))
        (IndexedSeq(res), axis)
      } else {
        (inputs.toIndexedSeq, axis)
      }
    }
  }

  // Reset before re-using the cell for another graph
  def reset(): Unit = {
    initCounter = -1
    counter = -1
  }

  /**
   * Construct symbol for one step of RNN.
   * @param inputs : sym.Variable input symbol, 2D, batch * num_units
   * @param states : sym.Variable state from previous step or begin_state().
   * @return  output symbol & state to next step of RNN.
   */
  def apply(inputs: Symbol, states: IndexedSeq[Symbol]): (Symbol, IndexedSeq[Symbol])

  // Parameters of this cell
  def params: RNNParams = {
    ownParams = false
    myParams
  }

  // shape and layout information of states
  // TODO
  def stateInfo: IndexedSeq[Option[RNNStateInfo]]

  // shapes of states
  def stateShape: IndexedSeq[Option[Shape]] = {
    stateInfo.map(_.map(_.shape))
  }

  // name(s) of gates
  def gateNames: IndexedSeq[String] = {
    IndexedSeq.empty[String]
  }

  /**
   * Initial state for this cell.
   * @param func default symbol.zeros
   *             Function for creating initial state. Can be symbol.zeros,
   *             symbol.uniform, symbol.Variable etc.
   *             Use symbol.Variable if you want to directly feed input as states.
   * @return nested list of Symbol starting states for first RNN step
   */
  def beginState(func: RNNStateInitFunction = new RNNStateInitFuncZeros): IndexedSeq[Symbol] = {
    require(!modified, "After applying modifier cells (e.g. DropoutCell) the base "
      + "cell cannot be called directly. Call the modifier cell instead.")
    stateInfo.map(info => {
      initCounter += 1
      val name = s"${prefix}begin_state_$initCounter"
      info.map(func.invoke(name, _)).getOrElse(func.invoke(name))
    })
  }

  /**
   * Unpack fused weight matrices into separate weight matrices
   * @param args dictionary containing packed weights. usually from Module.get_output()
   * @return dictionary with weights associated to this cell unpacked.
   */
  def unpackWeights(args: Map[String, NDArray]): Map[String, NDArray] = {
    if (gateNames == null || gateNames.isEmpty) {
      args
    } else {
      val newArgs = mutable.Map(args.toSeq: _*)
      val h = numHidden
      for (groupName <- Array("i2h", "h2h")) {
        val weight = newArgs.remove(s"$prefix${groupName}_weight")
        val bias = newArgs.remove(s"$prefix${groupName}_bias")
        gateNames.zipWithIndex.foreach { case (gate, j) =>
          val wname = s"$prefix$groupName${gate}_weight"
          newArgs.put(wname, weight.get.slice(j * h, (j + 1) * h).copy())
          val bname = s"$prefix$groupName${gate}_bias"
          newArgs.put(bname, bias.get.slice(j * h, (j + 1) * h).copy())
        }
      }
      newArgs.toMap
    }
  }

  /**
   * Pack separate weight matrices into fused weight.
   * @param args dictionary containing unpacked weights.
   * @return dictionary with weights associated to this cell packed.
   */
  def packWeights(args: Map[String, NDArray]): Map[String, NDArray] = {
    if (gateNames == null || gateNames.isEmpty) {
      args
    } else {
      val newArgs = mutable.Map(args.toSeq: _*)
      for (groupName <- Array("i2h", "h2h")) {
        val weight = gateNames.map(gate => {
          val wname = s"${this.prefix}$groupName${gate}_weight"
          require(newArgs.contains(wname), "Missing arg " + wname)
          newArgs.remove(wname).get
        })
        val bias = gateNames.map(gate => {
          val bname = s"${this.prefix}$groupName${gate}_bias"
          require(newArgs.contains(bname), "Missing arg " + bname)
          newArgs.remove(bname).get
        })
        newArgs.put(s"${this.prefix}${groupName}_weight", NDArray.concatenate(weight))
        newArgs.put(s"${this.prefix}${groupName}_bias", NDArray.concatenate(bias))
      }
      newArgs.toMap
    }
  }

  /**
   * Unroll an RNN cell across time steps.
   * @param length number of steps to unroll
   * @param inputs if inputs is a single Symbol (usually the output of Embedding symbol),
   *               it should have shape (batch_size, length, ...) if layout == 'NTC',
   *               or (length, batch_size, ...) if layout == 'TNC'.
   *               If inputs is a list of symbols (usually output of previous unroll),
   *               they should all have shape (batch_size, ...).
   * @param beginStateOpt input states. Created by begin_state() or output state of another cell.
   *                      Created from begin_state() if None.
   * @param layout layout of input symbol. Only used if inputs is a single Symbol.
   * @param mergeOutputs If False, return outputs as a list of Symbols.
   *                     If True, concatenate output across time steps
   *                     and return a single symbol with shape
   *                     (batch_size, length, ...) if layout == 'NTC',
   *                     or (length, batch_size, ...) if layout == 'TNC'.
   *                     If None, output whatever is faster
   * @return (output symbols, states)
   *
   */
  def unroll(inputs: Symbol*)(length: Int, beginStateOpt: Option[IndexedSeq[Symbol]] = None,
             layout: String = "NTC", mergeOutputs: Boolean = false)
    : (IndexedSeq[Symbol], IndexedSeq[Symbol]) = {
    this.reset()
    val (normalizedInputs, _) = normalizeSequence(inputs: _*)(length, layout, false)
    require(normalizedInputs.length == length)
    val beginState = beginStateOpt.getOrElse(this.beginState())

    var states = beginState
    val outputs = (0 until length).map(i => {
      val (output, newStates) = this(normalizedInputs(i), states)
      states = newStates
      output
    })

    val (newOutputs, _) = normalizeSequence(outputs: _*)(length, layout, mergeOutputs)

    (newOutputs, states)
  }
}

/**
 * Simple recurrent neural network cell
 * @param numHidden number of units in output symbol
 * @param activation type of activation function
 * @param prefix prefix for name of layers (and name of weight if params is None)
 * @param params container for weight sharing between cells. created if None.
 */
class RNNCell(override protected val numHidden: Int,
              protected val activation: String = "tanh",
              override protected val prefix: String = "rnn_",
              params: Option[RNNParams]) extends BaseRNNCell(prefix, params, numHidden) {
  require(params != None)
  private val paramsInst = params.get
  private val iW = paramsInst.get("i2h_weight")
  private val iB = paramsInst.get("i2h_bias")
  private val hW = paramsInst.get("h2h_weight")
  private val hB = paramsInst.get("h2h_bias")

  /**
   * Construct symbol for one step of RNN.
   * @param inputs : sym.Variable input symbol, 2D, batch * num_units
   * @param states : sym.Variable state from previous step or begin_state().
   * @return  output symbol & state to next step of RNN.
   */
  override def apply(inputs: Symbol, states: IndexedSeq[Symbol]): (Symbol, IndexedSeq[Symbol]) = {
    counter += 1
    val name = s"${prefix}t${counter}_"

    val i2h = Symbol.FullyConnected(name = s"${name}i2h")()(
      Map("data" -> inputs, "weight" -> iW, "bias" -> iB, "num_hidden" -> numHidden))
    val h2h = Symbol.FullyConnected(name = s"${name}h2h")()(
      Map("data" -> states(0), "weight" -> hW, "bias" -> hB, "num_hidden" -> numHidden))
    val output: Symbol = Symbol.Activation(name = s"${name}out")(i2h + h2h)(
      Map("act_type" -> activation))
    (output, IndexedSeq(output))
  }

  override def gateNames: IndexedSeq[String] = {
    IndexedSeq("")
  }

  // shape and layout information of states
  override def stateInfo: IndexedSeq[Option[RNNStateInfo]] = {
    val state = new RNNStateInfo(shape = Shape(0, numHidden), layout = "NC")
    IndexedSeq(Option(state))
  }
}
