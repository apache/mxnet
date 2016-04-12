package ml.dmlc.mxnet

import ml.dmlc.mxnet.optimizer.SGD
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class SerializerSuite extends FunSuite with BeforeAndAfterAll with Matchers {
  test("serialize and deserialize optimizer") {
    val optimizer: Optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0005f)
    val optSerialized: String = Serializer.encodeBase64String(
      Serializer.getSerializer.serialize(optimizer))
    assert(optSerialized.length > 0)

    val bytes = Serializer.decodeBase64String(optSerialized)
    val optDeserialized = Serializer.getSerializer.deserialize[Optimizer](bytes)

    assert(optDeserialized.isInstanceOf[SGD])
    val sgd = optDeserialized.asInstanceOf[SGD]

    val learningRate = classOf[SGD].getDeclaredField("learningRate")
    learningRate.setAccessible(true)
    assert(learningRate.get(sgd).asInstanceOf[Float] === 0.1f +- 1e-6f)

    val momentum = classOf[SGD].getDeclaredField("momentum")
    momentum.setAccessible(true)
    assert(momentum.get(sgd).asInstanceOf[Float] === 0.9f +- 1e-6f)

    val wd = classOf[SGD].getDeclaredField("wd")
    wd.setAccessible(true)
    assert(wd.get(sgd).asInstanceOf[Float] === 0.0005f +- 1e-6f)
  }
}
