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

import java.io.File

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
    var userDir : File = new File(System.getProperty("user.dir"))
    var nativeDir : File = new File(userDir, "init-native")
    if (!nativeDir.exists()) {
      nativeDir = new File(userDir.getParent, "init-native")
      if (!nativeDir.exists()) {
        throw new IllegalStateException("scala-init should be executed inside scala-package folder")
      }
    }
    val baseDir = nativeDir.getAbsolutePath

    val os = System.getProperty("os.name")
    // ref: http://lopica.sourceforge.net/os.html
    if (os.startsWith("Linux")) {
      System.load(s"$baseDir/target/libmxnet-init-scala.so")
    } else if (os.startsWith("Mac")) {
      System.load(s"$baseDir/target/libmxnet-init-scala.jnilib")
    } else {
      // TODO(yizhi) support windows later
      throw new UnsatisfiedLinkError()
    }
  }
}
