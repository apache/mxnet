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
 * \file PluginLayerHelper.java
 * \brief Helper class to generate layers using Caffe Plugin
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenerationHelper;
import io.mxnet.caffetranslator.Layer;

public class PluginLayerHelper {

    private final GenerationHelper gh;

    public PluginLayerHelper() {
        gh = new GenerationHelper();
    }

    public String getDataList(Layer layer) {
        StringBuilder sb = new StringBuilder();
        int index = 0;

        if (layer.getBottoms().size() == 0) {
            return null;
        }

        for (String bottom : layer.getBottoms()) {
            sb.append("data_" + index + "=" + gh.getVarname(bottom) + ", ");
            index++;
        }
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 2);
        }
        return sb.toString();
    }

    public String makeOneLine(String prototxt) {
        prototxt = prototxt.replaceAll("\n", "").replaceAll("\r", "");
        prototxt = prototxt.replaceAll("'", "\'");
        prototxt = prototxt.replaceAll("\\s{2,}", " ").trim();
        return prototxt;
    }

}
