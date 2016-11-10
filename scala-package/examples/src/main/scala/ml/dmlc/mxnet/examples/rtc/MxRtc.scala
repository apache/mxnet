package ml.dmlc.mxnet.examples.rtc

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Rtc

/**
 * @author Depeng Liang
 */
object MxRtc {
  private val logger = LoggerFactory.getLogger(classOf[MxRtc])

  def testKernel1(ctx: Context): Unit = {
      val x = NDArray.empty(ctx, 10)
      x.set(1f)
      val y = NDArray.empty(ctx, 10)
      y.set(2f)
      val rtc = new Rtc("abc", Array(("x", x)), Array(("y", y)), """
          __shared__ float s_rec[10];
          s_rec[threadIdx.x] = x[threadIdx.x];
          y[threadIdx.x] = expf(s_rec[threadIdx.x]*5.0);""")

      rtc.push(Array(x), Array(y), (1, 1, 1), (10, 1, 1))

      val result = y.toArray.zip(x.toArray).forall { case (l, r) =>
        Math.abs((l - Math.exp(r * 5.0))) < 1e-5f
      }
      assert(result)

      rtc.dispose()
  }

  def testKernel2(ctx: Context): Unit = {
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
      val zArr = z.toArray
      val result = zArr.indices.forall { idx =>
        Math.abs(zArr(idx) - Math.sqrt(xArr(idx) * yArr(idx) / 2.5f)) < 1e-7f
      }
      assert(result)

      rtc.dispose()
  }

  def main(args: Array[String]): Unit = {
    val mxtc = new MxRtc
    val parser: CmdLineParser = new CmdLineParser(mxtc)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(mxtc.gpu >= 0)

      val ctx = Context.gpu(mxtc.gpu)

      testKernel1(ctx)
      testKernel2(ctx)

   } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class MxRtc {
  @Option(name = "--gpu", usage = "which gpu card to use")
  private val gpu: Int = 0
}
