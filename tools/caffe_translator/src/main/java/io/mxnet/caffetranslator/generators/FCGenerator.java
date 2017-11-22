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
 * \file FCGenerator.java
 * \brief Generate fully connected layer
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.Map;

public class FCGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        StringBuilder out = new StringBuilder();

        ST st = getTemplate("fc");

        gh.fillNameDataAndVar(st, layer);

        gh.simpleFillTemplate(st, "num", layer, "inner_product_param.num_output", null);

        if (layer.attrEquals("inner_product_param.bias_term", "false")) {
            st.add("no_bias", "NoBiasPlease"); //value doesn't matter
        }

        String weightInit = gh.getInit(
                layer.getAttr("inner_product_param.weight_filler.type"),
                layer.getAttr("inner_product_param.weight_filler.value"));

        String biasInit = gh.getInit(
                layer.getAttr("inner_product_param.bias_filler.type"),
                layer.getAttr("inner_product_param.bias_filler.value"));

        if (weightInit != null || layer.getParams().size() >= 1) {
            Map<String, String> param = layer.getParams().get(0);
            out.append(
                    generateVar("weight", layer.getName() + "_weight",
                            param.get("param.lr_mult"), param.get("param.decay_mult"),
                            weightInit, null)
            );
            st.add("weight", "weight");
        }

        if (biasInit != null || layer.getParams().size() >= 2) {
            Map<String, String> param = layer.getParams().get(1);
            out.append(
                    generateVar("bias", layer.getName() + "_bias",
                            param.get("param.lr_mult"), param.get("param.decay_mult"),
                            biasInit, null)
            );
            st.add("bias", "bias");
        }

        out.append(st.render());

        return gh.makeGeneratorOutput(out.toString(), 1);
    }
}
