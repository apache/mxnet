package ml.dmlc.mxnet.examples.visualization

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import scala.util.parsing.json._
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Visualization

/**
 * @author Depeng Liang
 */
object ExampleVis {
  private val logger = LoggerFactory.getLogger(classOf[ExampleVis])

  val netsList = List("LeNet", "AlexNet", "VGG", "GoogleNet",
      "Inception_BN", "Inception_V3", "ResNet_Small")

  val netShapes = Map(
      "LeNet" -> Shape(1, 1, 28, 28),
      "AlexNet" -> Shape(1, 1, 224, 224),
      "VGG" -> Shape(1, 1, 224, 224),
      "GoogleNet" -> Shape(1, 1, 299, 299),
      "Inception_BN" -> Shape(1, 1, 299, 299),
      "Inception_V3" -> Shape(1, 1, 299, 299),
      "ResNet_Small" -> Shape(1, 1, 28, 28)
  )

  def getNetSymbol(net: String): (Symbol, Shape) = {
    assert(netsList.contains(net), s"Supported nets: ${netsList.mkString(", ")}")
    net match {
      case "LeNet" => (LeNet.getSymbol(), netShapes(net))
      case "AlexNet" => (AlexNet.getSymbol(), netShapes(net))
      case "VGG" => (VGG.getSymbol(), netShapes(net))
      case "GoogleNet" => (GoogleNet.getSymbol(), netShapes(net))
      case "Inception_BN" => (Inception_BN.getSymbol(), netShapes(net))
      case "Inception_V3" => (Inception_V3.getSymbol(), netShapes(net))
      case "ResNet_Small" => (ResNet_Small.getSymbol(), netShapes(net))
    }
  }

  def main(args: Array[String]): Unit = {
    val leis = new ExampleVis
    val parser: CmdLineParser = new CmdLineParser(leis)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(leis.outDir != null)

      val (sym, shape) = getNetSymbol(leis.net)

      val dot = Visualization.plotNetwork(symbol = sym,
          title = leis.net, shape = Map("data" -> shape),
          nodeAttrs = Map("shape" -> "rect", "fixedsize" -> "false"))

      dot.render(engine = "dot", format = "pdf", fileName = leis.net, path = leis.outDir)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ExampleVis {
  @Option(name = "--out-dir", usage = "the output path")
  private val outDir: String = null
  @Option(name = "--net", usage = "network to visualize, e.g. LeNet, AlexNet, VGG ...")
  private val net: String = "LeNet"
}
