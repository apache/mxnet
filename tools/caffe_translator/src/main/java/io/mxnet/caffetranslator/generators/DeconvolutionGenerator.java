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
 * \file DeconvolutionGenerator.java
 * \brief Generate Deconvolution layer
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.Map;

public class DeconvolutionGenerator extends BaseGenerator {

    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        StringBuilder out = new StringBuilder();
        ST st = getTemplate("deconvolution");
        gh.fillNameDataAndVar(st, layer);

        // Set kernel size
        gh.simpleFillTemplate(st, "kernel_h", layer, "convolution_param.kernel_h", null,
                "convolution_param.kernel_size");
        gh.simpleFillTemplate(st, "kernel_w", layer, "convolution_param.kernel_w", null,
                "convolution_param.kernel_size");

        // Set stride
        gh.simpleFillTemplate(st, "stride_h", layer, "convolution_param.stride_h", "1",
                "convolution_param.stride");
        gh.simpleFillTemplate(st, "stride_w", layer, "convolution_param.stride_w", "1",
                "convolution_param.stride");

        // Set padding
        gh.simpleFillTemplate(st, "pad_h", layer, "convolution_param.pad_h", "0",
                "convolution_param.pad");
        gh.simpleFillTemplate(st, "pad_w", layer, "convolution_param.pad_w", "0",
                "convolution_param.pad");

        // Use bias?
        if (layer.attrEquals("convolution_param.bias_term", "false")) {
            st.add("no_bias", "NoBiasPlease");
        }

        // Number of channels in output
        gh.simpleFillTemplate(st, "num_filter", layer, "convolution_param.num_output", null);

        // Group
        gh.simpleFillTemplate(st, "group", layer, "convolution_param.group", "PP_REMOVE");


        // Custom weight and bias if needed
        String weightInit = gh.getInit(
                layer.getAttr("convolution_param.weight_filler.type"),
                layer.getAttr("convolution_param.weight_filler.value"));

        String biasInit = gh.getInit(
                layer.getAttr("convolution_param.bias_filler.type"),
                layer.getAttr("convolution_param.bias_filler.value"));

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
        }

        out.append(st.render());
        return new GeneratorOutput(out.toString(), 1);
    }
}
