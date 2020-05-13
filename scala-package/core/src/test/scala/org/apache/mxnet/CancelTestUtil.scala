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

package org.apache.mxnet

import java.text.DecimalFormatSymbols
import java.util.Locale

import org.scalatest.Assertions

object CancelTestUtil {
  /**
   * Cancel the test if the system's locale uses a decimal separator other than '.'. Please see
   * #18097 for more information.
   */
  def assumeStandardDecimalSeparator(): Unit = {
    val lcNumeric = System.getenv("LC_NUMERIC");

    val decimalFormatSymbols = if (lcNumeric != null) {
      val localeName = lcNumeric.stripSuffix(".UTF-8".stripSuffix(".utf-8"))
      val locale = Locale.forLanguageTag(localeName)
      DecimalFormatSymbols.getInstance(locale)
    } else {
      DecimalFormatSymbols.getInstance()
    }

    val isStandardDecimalPoint = (decimalFormatSymbols.getDecimalSeparator == '.') &&
      (lcNumeric != null && lcNumeric.toLowerCase != "en_dk.utf-8") // Java doesn't seem to respect
                                                                    // the decimal separator
                                                                    // set in en_DK.UTF8, which is
                                                                    // used in CentOS CI jobs.
    if (!isStandardDecimalPoint) {
      Assertions.cancel("Some operators " +
        "break when the decimal separator is set to anything other than \".\". These operators " +
        "should be rewritten to utilize the new FFI. Please see #18097 for more information.")
    }
  }
}
