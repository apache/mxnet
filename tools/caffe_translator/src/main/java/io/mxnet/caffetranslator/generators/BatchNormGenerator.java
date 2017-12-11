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
 * \file BatchNormGenerator.java
 * \brief Generate BatchNorm layer
 */

package io.mxnet.caffetranslator.generators;

import io.mxnet.caffetranslator.GeneratorOutput;
import io.mxnet.caffetranslator.Layer;
import io.mxnet.caffetranslator.MLModel;
import org.stringtemplate.v4.ST;

public class BatchNormGenerator extends BaseGenerator {
    @Override
    public GeneratorOutput generate(Layer layer, MLModel model) {
        ST st = getTemplate("batchnorm");

        gh.fillNameDataAndVar(st, layer);

        if (layer.attrEquals("batch_norm_param.use_global_stats", "true")) {
            st.add("use_global_stats", true);
        }

        int layerIndex = layer.getLayerIndex();
        Layer nextLayer = model.getLayerList().get(layerIndex + 1);

        boolean nextLayerIsScale = false;
        if (nextLayer.getType().toLowerCase().equals("scale")) {
            String axis = nextLayer.getAttr("ScaleParameter.axis", "1");
            String numAxis = nextLayer.getAttr("ScaleParameter.num_axes", "1");
            if (axis.equals("1") && numAxis.equals("1")) {
                String biasTerm = nextLayer.getAttr("ScaleParameter.bias_term", "false");
                if (biasTerm.toLowerCase().equals("false")) {
                    nextLayerIsScale = true;
                }
            }
        }

        if (!nextLayerIsScale) {
            st.add("fix_beta", true);
            st.add("fix_gamma", true);
        }

        return new GeneratorOutput(st.render(), nextLayerIsScale ? 2 : 1);
    }
}
