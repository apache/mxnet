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

import scala.util.parsing.json._
import java.io.File
import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer

object Visualization {

  /**
   * A simplify implementation of the python-Graphviz library functionality
   * based on: https://github.com/xflr6/graphviz/tree/master/graphviz
   */
  class Dot(name: String) {
    // http://www.graphviz.org/cgi-bin/man?dot
    private val ENGINES = Set(
      "dot", "neato", "twopi", "circo", "fdp", "sfdp", "patchwork", "osage"
    )

    // http://www.graphviz.org/doc/info/output.html
    private val FORMATS = Set(
        "bmp",
        "canon", "dot", "gv", "xdot", "xdot1.2", "xdot1.4",
        "cgimage",
        "cmap",
        "eps",
        "exr",
        "fig",
        "gd", "gd2",
        "gif",
        "gtk",
        "ico",
        "imap", "cmapx",
        "imap_np", "cmapx_np",
        "ismap",
        "jp2",
        "jpg", "jpeg", "jpe",
        "pct", "pict",
        "pdf",
        "pic",
        "plain", "plain-ext",
        "png",
        "pov",
        "ps",
        "ps2",
        "psd",
        "sgi",
        "svg", "svgz",
        "tga",
        "tif", "tiff",
        "tk",
        "vml", "vmlz",
        "vrml",
        "wbmp",
        "webp",
        "xlib",
        "x11"
    )

    private val _head = "digraph %s{".format(name)
    private val _node = "\t%s %s"
    private val _edge = "\t\t%s -> %s %s"
    private val _tail = "}"
    private val _body = ArrayBuffer[String]()

    private def attribute(label: String = null, attrs: Map[String, String]): String = {
      if (label != null) {
        s"[label=$label ${("" /: attrs){ (acc, elem) => s"$acc ${elem._1}=${elem._2}"}}]"
      }
      else {
        s"[${("" /: attrs){ (acc, elem) => s"$acc ${elem._1}=${elem._2}"}}]"
      }
    }

    /**
     * Create a node.
     * @param name Unique identifier for the node inside the source.
     * @param label Caption to be displayed (defaults to the node name).
     * @param attrs Any additional node attributes (must be strings).
     */
    def node(name: String, label: String = null, attrs: Map[String, String]): Unit = {
      _body += _node.format(name, attribute(label, attrs))
    }

    /**
     * Create an edge between two nodes.
     * @param tailName Start node identifier.
     * @param headName End node identifier.
     * @param label Caption to be displayed near the edge.
     * @param attrs Any additional edge attributes (must be strings).
     */
    def edge(tailName: String, headName: String,
        label: String = null, attrs: Map[String, String]): Unit = {
      _body += _edge.format(tailName, headName, attribute(label, attrs))
    }

    private def save(filename: String, directory: String): String = {
      val path = s"$directory${File.separator}$filename"
      val writer = new PrintWriter(path)
      try {
        // scalastyle:off println
        writer.println(s"${this._head}")
        this._body.toArray.foreach { line => writer.println(s"$line") }
        writer.println(s"${this._tail}")
        writer.flush()
        // scalastyle:off println
      } finally {
        writer.close()
      }
      path
    }

    private def command(engine: String, format: String, filepath: String): String = {
      require(ENGINES.contains(engine) == true, s"unknown engine: $engine")
      require(FORMATS.contains(format) == true, s"unknown format: $format")
      s"$engine -T${format} -O $filepath"
    }

    /**
     * Render file with Graphviz engine into format.
     * @param engine The layout commmand used for rendering ('dot', 'neato', ...).
     * @param format The output format used for rendering ('pdf', 'png', ...).
     * @param fileName Name of the DOT source file to render.
     * @param path Path to save the Dot source file.
     */
    def render(engine: String = "dot", format: String = "pdf",
        fileName: String, path: String): Unit = {
      val filePath = this.save(fileName, path)
      val args = command(engine, format, filePath)
      import sys.process._
      try {
        args !
      } catch { case _ : Throwable =>
        val errorMsg = s"""failed to execute "$args", """ +
        """"make sure the Graphviz executables are on your systems' path"""
        throw new RuntimeException(errorMsg)
      }
    }
  }

  /**
   * convert shape string to list, internal use only
   * @param str shape string
   * @return list of string to represent shape
   */
  def str2Tuple(str: String): List[String] = {
    val re = """\d+""".r
    re.findAllIn(str).toList
  }

  /**
   * convert symbol to Dot object for visualization
   * @param symbol symbol to be visualized
   * @param title title of the dot graph
   * @param shape Map of shapes, str -> shape, given input shapes
   * @param nodeAttrs Map of node's attributes
   *               for example:
   *                      nodeAttrs = Map("shape" -> "oval", "fixedsize" -> "fasle")
   *                      means to plot the network in "oval"
   * @param hideWeights
   *               if true (default) then inputs with names like `*_weight`
   *               or `*_bias` will be hidden
   * @return Dot object of symbol
   */
  def plotNetwork(symbol: Symbol,
      title: String = "plot", shape: Map[String, Shape] = null,
      nodeAttrs: Map[String, String] = Map[String, String](),
      hideWeights: Boolean = true): Dot = {

    val (drawShape, shapeDict) = {
      if (shape == null) (false, null)
      else {
        val internals = symbol.getInternals()
        val (_, outShapes, _) = internals.inferShape(shape)
        require(outShapes != null, "Input shape is incomplete")
        val shapeDict = internals.listOutputs().zip(outShapes).toMap
        (true, shapeDict)
      }
    }
    val conf = JSON.parseFull(symbol.toJson) match {
      case None => null
      case Some(map) => map.asInstanceOf[Map[String, Any]]
    }
    require(conf != null)

    require(conf.contains("nodes"))
    val nodes = conf("nodes").asInstanceOf[List[Any]]

    // default attributes of node
    val nodeAttr = scala.collection.mutable.Map("shape" -> "box", "fixedsize" -> "true",
              "width" -> "1.3", "height" -> "0.8034", "style" -> "filled")
    // merge the dict provided by user and the default one
    nodeAttrs.foreach { case (k, v) => nodeAttr(k) = v }
    val dot = new Dot(name = title)
    // color map
    val cm = List(""""#8dd3c7"""", """"#fb8072"""", """"#ffffb3"""",
                  """"#bebada"""", """"#80b1d3"""", """"#fdb462"""",
                  """"#b3de69"""", """"#fccde5"""")

    // Internal helper to figure out if node should be hidden with hide_weights
    def looksLikeWeight(name: String): Boolean = {
      if (name.endsWith("_weight") || name.endsWith("_bias")) true
      else false
    }

    // make nodes
    val hiddenNodes = scala.collection.mutable.Set[String]()
    nodes.foreach { node =>
      val params = node.asInstanceOf[Map[String, Any]]
      val op = params("op").asInstanceOf[String]
      val name = params("name").asInstanceOf[String]
      val attrs = {
        if (params.contains("attr")) params("attr").asInstanceOf[Map[String, String]]
        else Map[String, String]()
      }
      // input data
      val attr = nodeAttr.clone()
      var label = name
      var continue = false
      op match {
        case "null" => {
          if (looksLikeWeight(name)) {
            if (hideWeights) hiddenNodes.add(name)
            continue = true
          }
          attr("shape") = "oval" // inputs get their own shape
          label = name
          attr("fillcolor") = cm(0)
        }
        case "Convolution" => {
          val kernel = str2Tuple(attrs("kernel"))
          val stride = if (attrs.contains("stride")) str2Tuple(attrs("stride")) else List(1)
          label =
            """"Convolution\n%s/%s, %s"""".format(
              kernel.mkString("x"), stride.mkString("x"), attrs("num_filter"))
          attr("fillcolor") = cm(1)
        }
        case "FullyConnected" => {
          label = s""""FullyConnected\n${attrs("num_hidden")}""""
          attr("fillcolor") = cm(1)
        }
        case "BatchNorm" => attr("fillcolor") = cm(3)
        case "Activation" | "LeakyReLU" => {
          label = s""""${op}\n${attrs("act_type")}""""
          attr("fillcolor") = cm(2)
        }
        case "Pooling" => {
          val kernel = str2Tuple(attrs("kernel"))
          val stride = if (attrs.contains("stride")) str2Tuple(attrs("stride")) else List(1)
          label =
            s""""Pooling\n%s, %s/%s"""".format(
              attrs("pool_type"), kernel.mkString("x"), stride.mkString("x"))
          attr("fillcolor") = cm(4)
        }
        case "Concat" | "Flatten" | "Reshape" => attr("fillcolor") = cm(5)
        case "Softmax" => attr("fillcolor") = cm(6)
        case _ => {
          attr("fillcolor") = cm(7)
          if (op == "Custom") label = attrs("op_type")
        }
      }
      if (!continue) dot.node(name = name , label, attr.toMap)
    }

    val outIdx = scala.collection.mutable.Map[String, Int]()
    // add edges
    nodes.foreach { node =>
      val params = node.asInstanceOf[Map[String, Any]]
      val op = params("op").asInstanceOf[String]
      val name = params("name").asInstanceOf[String]
      if (op != "null") {
        val inputs = params("inputs").asInstanceOf[List[List[Double]]]
        for (item <- inputs) {
          val inputNode = nodes(item(0).toInt).asInstanceOf[Map[String, Any]]
          val inputName = inputNode("name").asInstanceOf[String]
          if (!hiddenNodes.contains(inputName)) {
            val attrs = scala.collection.mutable.Map("dir" -> "back", "arrowtail" -> "open")
            // add shapes
            if (drawShape) {
              val key = {
                if (inputNode("op").asInstanceOf[String] != "null") {
                  var key = s"${inputName}_output"
                  if (inputNode.contains("attr")) {
                    val params = inputNode("attr").asInstanceOf[Map[String, String]]
                    if (params.contains("num_outputs")) {
                      if (!outIdx.contains(name)) outIdx(name) = params("num_outputs").toInt - 1
                      key += outIdx(name)
                      outIdx(name) = outIdx(name) - 1
                    }
                  }
                  key
                } else inputName
              }
              val shape = shapeDict(key).toArray.drop(1)
              val label = s""""${shape.mkString("x")}""""
              attrs("label") = label
            }
            dot.edge(tailName = name, headName = inputName, attrs = attrs.toMap)
          }
        }
      }
    }
    dot
  }
}
