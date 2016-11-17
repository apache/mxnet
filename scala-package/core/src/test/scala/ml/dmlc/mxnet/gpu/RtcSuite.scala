package ml.dmlc.mxnet

import org.scalatest.{Ignore, BeforeAndAfterAll, FunSuite}

@Ignore
class RtcSuite extends FunSuite with BeforeAndAfterAll {
  test("test kernel 1") {
    val ctx = Context.gpu(0)
    val x = NDArray.empty(ctx, 10)
    x.set(1f)
    val y = NDArray.empty(ctx, 10)
    y.set(2f)
    val rtc = new Rtc("abc", Array(("x", x)), Array(("y", y)), """
        __shared__ float s_rec[10];
        s_rec[threadIdx.x] = x[threadIdx.x];
        y[threadIdx.x] = expf(s_rec[threadIdx.x]*5.0);""")

    rtc.push(Array(x), Array(y), (1, 1, 1), (10, 1, 1))

    val gt = x.toArray.map( x => Math.exp(x * 5.0).toFloat )

    rtc.dispose()
    assert(CheckUtils.reldiff(y.toArray, gt) < 1e-5f)
  }

  test("test kernel 2") {
    val ctx = Context.gpu(0)
    val x = NDArray.empty(ctx, 33554430)
    x.set(1f)
    val y = NDArray.empty(ctx, 33554430)
    y.set(2f)
    val z = NDArray.empty(ctx, 33554430)

    val rtc = new Rtc("multiplyNumbers", Array(("x", x), ("y", y)), Array(("z", z)), """
      int tid = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;
      z[tid] = sqrt(x[tid] * y[tid] / 2.5);""")

    rtc.push(Array(x, y), Array(z), (128, 1024, 1), (256, 1, 1))

    val xArr = x.toArray
    val yArr = y.toArray
    val gt = xArr.indices.map( i => Math.sqrt(xArr(i) * yArr(i) / 2.5f).toFloat )

    rtc.dispose()
    assert(CheckUtils.reldiff(z.toArray, gt.toArray) < 1e-7f)
  }
}
