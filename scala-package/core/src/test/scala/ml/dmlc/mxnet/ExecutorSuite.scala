package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import ml.dmlc.mxnet.CheckUtils._

class ExecutorSuite extends FunSuite with BeforeAndAfterAll {
  test("bind") {
    val shape = Shape(100, 30)
    val lhs = Symbol.Variable("lhs")
    val rhs = Symbol.Variable("rhs")
    val ret = lhs + rhs
    assert(ret.listArguments().toArray === Array("lhs", "rhs"))

    val lhsArr = Random.uniform(-10f, 10f, shape)
    val rhsArr = Random.uniform(-10f, 10f, shape)
    val lhsGrad = NDArray.empty(shape)
    val rhsGrad = NDArray.empty(shape)

    val executor = ret.bind(Context.cpu(), args = Seq(lhsArr, rhsArr),
                            argsGrad = Seq(lhsGrad, rhsGrad))
    val exec3 = ret.bind(Context.cpu(), args = Seq(lhsArr, rhsArr))
    val exec4 = ret.bind(Context.cpu(), args = Map("rhs" -> rhsArr, "lhs" -> lhsArr),
                         argsGrad = Map("lhs" -> lhsGrad, "rhs" -> rhsGrad))
    executor.forward()
    exec3.forward()
    exec4.forward()

    val out1 = lhsArr + rhsArr
    val out2 = executor.outputs(0)
    val out3 = exec3.outputs(0)
    val out4 = exec4.outputs(0)
    assert(reldiff(out1, out2) < 1e-6)
    assert(reldiff(out1, out3) < 1e-6)
    assert(reldiff(out1, out4) < 1e-6)

    // test gradient
    val outGrad = NDArray.ones(shape)
    val (lhsGrad2, rhsGrad2) = (outGrad, outGrad)
    executor.backward(Array(outGrad))
    assert(reldiff(lhsGrad, lhsGrad2) < 1e-6)
    assert(reldiff(rhsGrad, rhsGrad2) < 1e-6)
  }

  test("reshape") {
    val x = Symbol.Variable("x")
    val y = Symbol.FullyConnected()()(Map("data" -> x, "num_hidden" -> 4))

    val exec = y.simpleBind(Context.cpu(), "write", shapeDict = Map("x" -> Shape(5, 4)))
    exec.argArrays(0).set(1)
    exec.argArrays(1).set(1)
    exec.argArrays(2).set(0)

    val newExec = exec.reshape(kwargs = Map("x" -> Shape(3, 4)))
    newExec.forward(isTrain = false)
    // test sub exec forward
    assert(newExec.outputs(0).toArray.forall(_ == 4))
    // test shared memory
    assert(exec.outputs(0).toArray.take(3).forall(_ == 4))
    // test base exec forward
    exec.forward(isTrain = false)
    assert(exec.outputs(0).toArray.forall(_ == 4))
  }
}
