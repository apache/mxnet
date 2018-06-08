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

package org.apache.mxnet.init

object Base {
  tryLoadInitLibrary()
  val _LIB = new LibInfo

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type CPtrAddress = Long

  type NDArrayHandle = CPtrAddress
  type FunctionHandle = CPtrAddress
  type KVStoreHandle = CPtrAddress
  type ExecutorHandle = CPtrAddress
  type SymbolHandle = CPtrAddress

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadInitLibrary(): Unit = {
    var baseDir = System.getProperty("user.dir") + "/init-native"
    // TODO(lanKing520) Update this to use relative path to the MXNet director.
    // TODO(lanking520) baseDir = sys.env("MXNET_BASEDIR") + "/scala-package/init-native"
    if (System.getenv().containsKey("MXNET_BASEDIR")) {
      baseDir = sys.env("MXNET_BASEDIR")
    }
    val os = System.getProperty("os.name")
    // ref: http://lopica.sourceforge.net/os.html
    if (os.startsWith("Linux")) {
      System.load(s"$baseDir/linux-x86_64/target/libmxnet-init-scala-linux-x86_64.so")
    } else if (os.startsWith("Mac")) {
      System.load(s"$baseDir/osx-x86_64/target/libmxnet-init-scala-osx-x86_64.jnilib")
    } else {
      // TODO(yizhi) support windows later
      throw new UnsatisfiedLinkError()
    }
  }
}
