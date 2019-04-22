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

  /**
    * This C Pointer Address point to the
    * actual memory piece created in MXNet Engine
    */
  type CPtrAddress = Long

  /**
    * NDArrayHandle is the C pointer to
    * the NDArray
    */
  type NDArrayHandle = CPtrAddress
  /**
    * FunctionHandle is the C pointer to
    * the ids of the operators
    */
  type FunctionHandle = CPtrAddress
  /**
    * KVStoreHandle is the C pointer to
    * the KVStore
    */
  type KVStoreHandle = CPtrAddress
  /**
    * ExecutorHandle is the C pointer to
    * the Executor
    */
  type ExecutorHandle = CPtrAddress
  /**
    * SymbolHandle is the C pointer to
    * the Symbol
    */
  type SymbolHandle = CPtrAddress

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadInitLibrary(): Unit = {
    val userDir : File = new File(System.getProperty("user.dir"))
    var nativeDir : File = new File(userDir, "init-native")
    if (!nativeDir.exists()) {
      nativeDir = new File(userDir.getParent, "init-native")
      if (!nativeDir.exists()) {
        throw new IllegalStateException("scala-init should be executed inside scala-package folder")
      }
    }
    val baseDir = nativeDir.getAbsolutePath

    val os = System.getProperty("os.name")
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
