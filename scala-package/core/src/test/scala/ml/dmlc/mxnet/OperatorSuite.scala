package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base.Shape
import ml.dmlc.mxnet.CheckUtils._

import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}
import org.scalacheck.Gen

import scala.collection.mutable

class OperatorSuite extends FunSuite with BeforeAndAfterAll
  with Matchers with GeneratorDrivenPropertyChecks {
  private def checkElementwiseSumWithShape(shape: Shape, n: Int) = {
    // forward
    val inputs = (0 until n).map(i => Symbol.Variable(s"arg $i"))
    val out = Symbol.ElementWiseSum("esum", inputs: _*)
    val arr = (0 until n).map(_ => Random.uniform(-10, 10, shape))
    val arrGrad = (0 until n).map(_ => NDArray.empty(shape))
    val exec = out.bind(Context.cpu(), args = arr, argsGrad = arrGrad)
    exec.forward()
    val forwardOutput = exec.outputs(0)
    val forwardOutputExpected = arr.reduce(_ + _)
    assert(reldiff(forwardOutput, forwardOutputExpected) < 1e-6)

    // backward
    val outGrad = Random.uniform(-10, 10, shape)
    exec.backward(outGrad)
    arrGrad.foreach(grad => assert(grad === outGrad))
  }

  test("elementwise sum") {
    checkElementwiseSumWithShape(Vector(5, 5, 3), 4)
    forAll (Gen.choose(1, 4), Gen.choose(1, 8)) { (dim, n) =>
      forAll (Gen.listOfN(dim, Gen.choose(1, Math.pow(1000, 1.0 / dim).toInt))) { shape =>
        checkElementwiseSumWithShape(shape.toVector, n)
      }
    }
  }

  // TODO: checkSliceChannel

  private def checkConcatWithShape(shapes: Seq[Shape], dimension: Int, skipSecond: Boolean) = {
    // if skipSecond is true, second argument will not have gradient.
    // it is to test #1130
    // forward
    val targetDim = shapes.map(_(dimension)).sum

    val inputs = (0 until shapes.size).map(i => Symbol.Variable(s"arg$i"))
    val out = Symbol.Concat(inputs, Map("name" -> "conc", "dim" -> dimension))
    val arr = shapes.map { shape =>
      val nd = NDArray.empty(shape)
      nd.set(shape(dimension))
    }
    val arrNp = arr.map(_.copy())
    val arrGrad = shapes.map(NDArray.empty(_))
    val argNames = out.listArguments()
    val dictGrad =
      (argNames zip arrGrad).filter { case (name, d) =>
        !skipSecond || name != "arg1"
      }.toMap

    val args = out.listArguments()
    val (argShapes, outShapes, auxShapes) = out.inferShape(args.zip(shapes).toMap)
    val outGrad = NDArray.empty(outShapes(0))
    val exec1 = out.bind(Context.cpu(), arr, dictGrad)
    exec1.forward()
    val out1 = exec1.outputs(0)
    // FIXME: only support concatenate at axis0
    val ret = NDArray.concatenate(arr)
    assert(out1 === ret)

    // backward
    out1.copyTo(outGrad)
    outGrad += 1
    exec1.backward(outGrad)
    argNames.zipWithIndex.foreach { case (name, i) =>
      if (!skipSecond || name != "arg1") {
        val grad = dictGrad(name)
        val npGrad = arrNp(i)
        assert(grad === npGrad + 1)
      }
    }
  }

  test("concat") {
    val merge = Array(2, 3, 4, 5, 6)
    forAll (Gen.choose(2, 5)) { dim =>
      val shapes = mutable.ArrayBuffer.empty[Vector[Int]]
      for (i <- 0 until dim) {
        shapes += Vector(merge(i), 2)
      }
      // TODO: check dimension > 0
      checkConcatWithShape(shapes, 0, skipSecond = true)
      checkConcatWithShape(shapes, 0, skipSecond = false)
    }
  }

  private def checkRegression(model: Symbol,
                              forward: Float => Float,
                              backward: (Float, Float) => Float) = {
    val shape = Vector(3, 1)
    val arrData = Random.uniform(-1, 1, shape)
    val arrLabel = Random.uniform(0, 1, Vector(shape.head))
    val arrGrad = NDArray.empty(shape)
    val exec1 = model.bind(Context.cpu(),
      args = Array(arrData, arrLabel), argsGrad = Map("data" -> arrGrad))
    exec1.forward()
    assert(exec1.outputs(0).shape === shape)
    val out1 = exec1.outputs(0).toArray
    val npout = arrData.toArray.map(forward(_))
    assert(CheckUtils.reldiff(npout, out1) < 1e-6f)

    exec1.backward()
    // arrData shape: Vector(3, 1)
    // arrLabel shape: Vector(3)
    val npoutBack = (npout zip arrLabel.toArray).map { case (data, label) =>
      backward(data, label)
    }
    assert(CheckUtils.reldiff(npoutBack, arrGrad.toArray) < 1e-6f)
  }

  test("regression") {
    checkRegression(Symbol.LogisticRegressionOutput(
      Array(Symbol.Variable("data"), Symbol.Variable("label"))),
      (x: Float) => 1.0f / (1.0f + Math.exp(-x).toFloat),
      (x: Float, y: Float) => x - y)
    checkRegression(Symbol.LinearRegressionOutput(
      Array(Symbol.Variable("data"), Symbol.Variable("label"))),
      (x: Float) => x,
      (x: Float, y: Float) => x - y)
  }

  // TODO: test softmax

  test("swap axes") {
    val data = Symbol.Variable("data")
    val shape = Vector(2, 3, 4)
    val arrData = NDArray.ones(shape)
    arrData.slice(0).set(1f)
    arrData.slice(1).set(2f)
    // arrData =
    //
    // [[[ 1.,  1.,  1.,  1.],
    //   [ 1.,  1.,  1.,  1.],
    //   [ 1.,  1.,  1.,  1.]],
    //
    // [[ 2.,  2.,  2.,  2.],
    //  [ 2.,  2.,  2.,  2.],
    //  [ 2.,  2.,  2.,  2.]]]
    val swap0 = Symbol.SwapAxis(data = data, dim1 = 0, dim2 = 2)
    val swap = Symbol.SwapAxis(data = swap0, dim1 = 1, dim2 = 2)
    val exec = swap.bind(Context.cpu(), args = Array(arrData))
    exec.forward()
    val out = exec.outputs(0)

    // After swapaxes(swapaxes(arrData, 0, 2), 1, 2)
    // out should be
    // [[[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]],
    //
    //  [[ 1.,  1.,  1.],
    //   [ 2.,  2.,  2.]]]
    assert(out.shape === Vector(4, 2, 3))
    for (i <- 0 until 4) {
      val axis0 = out.slice(i)
      assert(CheckUtils.reldiff(axis0.toArray, Array(1f, 1f, 1f, 2f, 2f, 2f)) < 1e-6f)
    }
  }

  test("scalar op") {
    val data = Symbol.Variable("data")
    val shape = Vector(3, 4)
    val dataTmp = NDArray.ones(shape) * 5

    val test = {
      import ml.dmlc.mxnet.SymbolConversions._
      2 / (4 - ((1 + data + 1) * 2 / 5) - 0.2)
    }

    val (npout1, npout) = {
      import ml.dmlc.mxnet.NDArrayConversions._
      val npout1 = 4 - ((1 + dataTmp + 1) * 2 / 5) - 0.2f
      val npout = 2 / npout1
      (npout1, npout)
    }

    checkSymbolicForward(test, Array(dataTmp), Array(npout))

    val npoutGrad = new NDArrayConversions(2f * (2f * 2f / 5f)) / (npout1 * npout1)

    checkSymbolicBackward(test, Array(dataTmp), Array(NDArray.ones(shape) * 2), Array(npoutGrad))
  }

  test("scalar pow") {
    val data = Symbol.Variable("data")
    val shape = Vector(1, 1)
    val dataTmp = NDArray.ones(shape) * 3
    val dataTmpPowered = NDArray.ones(shape) * 9
    val test = Symbol.pow(data, 2)
    // TODO: check_numeric_gradient(test, [data_tmp])
    checkSymbolicForward(test, Array(dataTmp), Array(dataTmpPowered))
    checkSymbolicBackward(test, Array(dataTmp), Array(NDArray.ones(shape)), Array(dataTmp * 2))
  }

  test("symbol pow") {
    val shape = Vector(1, 1)

    val data = Symbol.Variable("data")
    val dataTmp = NDArray.ones(shape) * 2

    val exp = Symbol.Variable("exp")
    val expTmp = NDArray.ones(shape) * 3

    val test = Symbol.pow(data, exp)

    // TODO: check_numeric_gradient(test, [data_tmp, exp_tmp])
    checkSymbolicForward(test, Seq(dataTmp, expTmp), Seq(NDArray.ones(shape) * 8))

    val dataDir = NDArray.ones(shape) * 4 * expTmp // dataTmp**(expTmp - 1) * expTmp
    // expDir = dataTmp**(expTmp) * log(dataTmp)
    val expDir = NDArray.ones(shape) * 8 * (NDArray.ones(shape) * Math.log(2).toFloat)
    checkSymbolicBackward(test, Seq(dataTmp, expTmp),
                          Seq(NDArray.ones(shape)), Seq(dataDir, expDir))
  }

  test("pow fn") {
    val shape = Vector(3, 4)
    val exp = Symbol.Variable("exp")
    val y = Symbol.pow(2, exp)
    val x = NDArray.ones(shape) * 3
    // TODO: check_numeric_gradient(y, [x])
    checkSymbolicForward(y, Seq(x), Seq(NDArray.ones(shape) * 8)) // 2**x
    checkSymbolicBackward(y, Seq(x), Seq(NDArray.ones(shape)),
      // log(2) * 2**x
      Seq(NDArray.ones(shape) * 8 * Math.log(2).toFloat))
  }

  /**
   * Compare foward call to expected value.
   * @param sym output symbol
   * @param location list of numpy arrays corresponding to sym.list_arguments
   * @param expected list of arrays corresponding to sym.outputs
   * @param checkEps relative error to check to
   */
  private def checkSymbolicForward(sym: Symbol,
                                   location: Seq[NDArray],
                                   expected: Seq[NDArray],
                                   checkEps: Float = 1e-5f): Unit = {
    val arrData = location.map(_.copy())
    val arrGrad = location.map(array => NDArray.empty(array.shape))

    val executor = sym.bind(Context.cpu(), args = arrData, argsGrad = arrGrad)

    val inps = executor.argArrays
    assert(inps.size === location.size,
      s"Executor argArrays and and location len do not match." +
      s"Got ${inps.size} inputs and ${location.size} locations")

    for ((inp, source) <- location zip executor.argArrays) {
      source.set(inp)
    }
    for (g <- executor.gradArrays) {
      if (g != null) {
        g.set(0f)
      }
    }

    assert(executor.outputs.length === 1)

    executor.forward()

    for ((expect, output) <- expected zip executor.outputs) {
      assert(reldiff(expect, output) <= checkEps)
    }
  }

  /**
   * Compare backwards call to expected value.
   * @param sym output symbol
   * @param location list of numpy arrays corresponding to sym.list_arguments
   * @param grad list of numpy arrays corresponding to sym.outputs for incoming gradient
   * @param expected list of arrays corresponding to sym.outputs
   * @param checkEps relative error to check to
   */
  private def checkSymbolicBackward(sym: Symbol,
                                    location: Seq[NDArray],
                                    grad: Seq[NDArray],
                                    expected: Seq[NDArray],
                                    checkEps: Float = 1e-5f): Unit = {
    val arrData = location.map(_.copy())
    val arrGrad = location.map(array => NDArray.empty(array.shape))
    val outGrad = grad.map(_.copy()).toArray

    val executor = sym.bind(Context.cpu(), args = arrData, argsGrad = arrGrad)

    val inps = executor.argArrays
    assert(inps.size === location.size,
      s"Executor argArrays and and location len do not match." +
        s"Got ${inps.size} inputs and ${location.size} locations")
    for ((inp, source) <- location zip executor.argArrays) {
      source.set(inp)
    }
    for (g <- executor.gradArrays) {
      if (g != null) {
        g.set(0f)
      }
    }

    executor.forward()
    executor.backward(outGrad)

    for ((expect, grad) <- expected zip executor.gradArrays) {
      assert(reldiff(expect, grad) <= checkEps)
    }
  }
}
