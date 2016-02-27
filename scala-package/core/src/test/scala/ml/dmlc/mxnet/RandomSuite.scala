package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class RandomSuite extends FunSuite with BeforeAndAfterAll {
  test("uniform on cpu") {
    Context.cpu().withScope {
      val (a, b) = (-10, 10)
      val shape = Shape(100, 100)
      Random.seed(128)
      val un1 = Random.uniform(a, b, shape)
      Random.seed(128)
      val un2 = Random.uniform(a, b, shape)
      assert(un1 === un2)
      assert(Math.abs(un1.toArray.sum / un1.size - (a + b) / 2f) < 0.1)
    }
  }

  test("normal on cpu") {
    val (mu, sigma) = (10f, 2f)
    val shape = Shape(100, 100)
    Random.seed(128)
    val ret1 = Random.normal(mu, sigma, shape)
    Random.seed(128)
    val ret2 = Random.normal(mu, sigma, shape)
    assert(ret1 === ret2)

    val array = ret1.toArray
    val mean = array.sum / ret1.size
    val devs = array.map(score => (score - mean) * (score - mean))
    val stddev = Math.sqrt(devs.sum / ret1.size)

    assert(Math.abs(mean - mu) < 0.1)
    assert(Math.abs(stddev - sigma) < 0.1)
  }
}
