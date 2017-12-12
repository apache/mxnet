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
 * \file AccuracyMetricsGenerator.java
 * \brief Generate Accuracy metric
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GenerationHelper;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

import java.util.HashMap;
import java.util.Map;

public class AccuracyMetricsGenerator {

    private final Map<String, String> map;
    private final GenerationHelper gh;

    public AccuracyMetricsGenerator() {
        map = new HashMap<>();
        gh = new GenerationHelper();
    }

    public String generate(MLModel model) {
        StringBuilder out = new StringBuilder();
        generateMap(model);

        for (Layer layer : model.getLayerList()) {
            if (layer.getType().equals("Accuracy")) {
                ST st;
                if (layer.getAttr("accuracy_param.top_k", "1").equals("1")) {
                    st = gh.getTemplate("accuracy");
                } else {
                    st = gh.getTemplate("top_k_accuracy");
                    st.add("k", layer.getAttr("accuracy_param.top_k"));
                }

                st.add("var", gh.getVarname(layer.getTop()));
                String outputName = map.get(layer.getBottoms().get(0)) + "_output";
                st.add("output_name", outputName);
                st.add("label_name", layer.getBottoms().get(1));
                st.add("name", layer.getName());

                out.append(st.render());
                out.append(System.lineSeparator());
            }
        }

        return out.toString();
    }

    private void generateMap(MLModel model) {
        for (Layer layer : model.getLayerList()) {
            // If this is not SoftmaxWithLoss, move on
            if (!layer.getType().equals("SoftmaxWithLoss")) {
                continue;
            }

            map.put(layer.getBottoms().get(0), layer.getName());
        }
    }
}
