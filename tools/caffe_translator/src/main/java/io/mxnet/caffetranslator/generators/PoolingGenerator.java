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
 * \file PoolingGenerator.java
 * \brief Generate Pooling layer
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class PoolingGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("pooling");

        gh.fillNameDataAndVar(st, layer);

        boolean globalPooling = layer.getAttr("pooling_param.global_pooling", "false")
                .toLowerCase().equals("true");

        if (globalPooling) {
            st.add("global_pool", "True");
            st.add("kernel_h", "1");
            st.add("kernel_w", "1");
        } else {
            // Set kernel size
            gh.simpleFillTemplate(st, "kernel_h", layer, "pooling_param.kernel_h", null,
                    "pooling_param.kernel_size");
            gh.simpleFillTemplate(st, "kernel_w", layer, "pooling_param.kernel_w", null,
                    "pooling_param.kernel_size");
        }

        // Set stride
        gh.simpleFillTemplate(st, "stride_h", layer, "pooling_param.stride_h", "1",
                "pooling_param.stride");
        gh.simpleFillTemplate(st, "stride_w", layer, "pooling_param.stride_w", "1",
                "pooling_param.stride");

        // Set padding
        gh.simpleFillTemplate(st, "pad_h", layer, "pooling_param.pad_h", "0",
                "pooling_param.pad");
        gh.simpleFillTemplate(st, "pad_w", layer, "pooling_param.pad_w", "0",
                "pooling_param.pad");

        // Set type
        String poolType = layer.getAttr("pooling_param.pool");
        switch (poolType) {
            case "MAX":
                st.add("type", "max");
                break;
            case "AVE":
                st.remove("type");
                st.add("type", "avg");
                break;
            case "STOCHASTIC":
                System.err.println("Stochastic pooling type not supported.");
                st.add("type", "???");
                break;
        }

        return new GeneratorOutput(st.render(), 1);
    }

}
