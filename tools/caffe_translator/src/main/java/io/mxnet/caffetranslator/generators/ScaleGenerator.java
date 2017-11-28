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
 * \file ScaleGenerator.java
 * \brief Generate Scale layer
 */


package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;

public class ScaleGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        PluginIntLayerGenerator generator = new PluginIntLayerGenerator();

        boolean use_bias = layer.getAttr("scale_param.bias_term", "false").toLowerCase().equals("true");

        StringBuilder out = new StringBuilder();

        if (use_bias) {
            out.append(generator.generate(layer, model, 2).code);
        } else {
            out.append(generator.generate(layer, model, 1).code);
        }

        String fillerType = layer.getAttr("filler.type");
        String fillerValue = layer.getAttr("filler.value");
        if (fillerType == null && fillerValue == null) {
            fillerValue = "1";
        }
        out.append(gh.initializeParam(gh.getVarname(layer.getTop()), 1, gh.getInit(fillerType, fillerValue)));

        if (use_bias) {
            fillerType = layer.getAttr("bias_filler.type");
            fillerValue = layer.getAttr("bias_filler.value");
            if (fillerType == null && fillerValue == null) {
                fillerValue = "0";
            }
            out.append(gh.initializeParam(gh.getVarname(layer.getTop()), 2, gh.getInit(fillerType, fillerValue)));
        }

        return gh.makeGeneratorOutput(out.toString(), 1);
    }
}
