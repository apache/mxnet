/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Utils.java
 * \brief General util functions
 */

package io.mxnet.caffetranslator;

import java.util.Collections;

public class Utils {
    public static String indent(String str, int level, boolean useSpaces, int numSpaces) {
        String prefix;
        if (!useSpaces) {
            prefix = String.join("", Collections.nCopies(level, "\t"));
        } else {
            String spaces = String.join("", Collections.nCopies(numSpaces, " "));
            prefix = String.join("", Collections.nCopies(level, spaces));
        }

        String indented = str.replaceAll("(?m)^", prefix);
        return indented;
    }
}
