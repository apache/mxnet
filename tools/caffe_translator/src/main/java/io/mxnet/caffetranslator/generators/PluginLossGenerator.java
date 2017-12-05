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
 * \file PluginLossGenerator.java
 * \brief Generate loss layer using Caffe Plugin
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class PluginLossGenerator extends BaseGenerator {

    private final PluginLayerHelper helper;

    public PluginLossGenerator() {
        super();
        helper = new PluginLayerHelper();
    }

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("CaffePluginLossLayer");

        st.add("name", layer.getName());

        // Handle data
        if (layer.getBottoms().size() != 1) {
            st.add("num_data", layer.getBottoms().size());
        }
        String dataList = helper.getDataList(layer);
        st.add("data", dataList);

        // Set prototxt
        String prototxt = helper.makeOneLine(layer.getPrototxt());
        st.add("prototxt", prototxt);

        // Handle multiple outputs
        if (layer.getTops().size() > 1) {
            st.add("tops", layer.getTops());
            st.add("var", "out");
        } else if (layer.getTops().size() == 1) {
            st.add("var", layer.getTop());
        }

        return new GeneratorOutput(st.render(), 1);
    }

}
