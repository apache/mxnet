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
