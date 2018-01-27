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

package ml.dmlc.mxnet.spark.utils

import java.io.IOException
import java.net.{ServerSocket, NetworkInterface}
import java.util.regex.Pattern

/**
 * Helper functions to decide ip address / port
 * @author Yizhi
 */
object Network {
  private val IPADDRESS_PATTERN = Pattern.compile(
    "^([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\." +
      "([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\." +
      "([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\." +
      "([01]?\\d\\d?|2[0-4]\\d|25[0-5])$")

  def ipAddress: String = {
    val interfaces = NetworkInterface.getNetworkInterfaces
    while (interfaces.hasMoreElements) {
      val interface = interfaces.nextElement
      val addresses = interface.getInetAddresses
      while (addresses.hasMoreElements) {
        val address = addresses.nextElement
        val ip = address.getHostAddress
        if (!ip.startsWith("127.") && IPADDRESS_PATTERN.matcher(ip).matches()) {
          return ip
        }
      }
    }
    "127.0.0.1"
  }

  def availablePort: Int = {
    try {
      val serverSocket = new ServerSocket(0)
      val port = serverSocket.getLocalPort
      try {
        serverSocket.close()
      } catch {
        case _: IOException => // do nothing
      }
      port
    } catch {
      case ex: Throwable => throw new IOException("Cannot find an available port")
    }
  }
}
