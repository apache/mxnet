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
 * \file MLModel.java
 * \brief Models a ML model
 */

package io.mxnet.caffetranslator;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MLModel {

    public MLModel() {
        layerList = new ArrayList<>();
        layerLookup = new HashMap<>();
        layerIndex = 0;
    }

    @Getter
    @Setter
    private String name;

    @Getter
    @Setter
    private List<Layer> layerList;

    private final Map<String, Map<String, Layer>> layerLookup;

    private int layerIndex;

    public void addLayer(Layer layer) {

        layer.setLayerIndex(layerIndex++);
        layerList.add(layer);

        String name = layer.getName();
        String includePhase = layer.getAttr("include.phase");
        includePhase = (includePhase == null) ? "" : includePhase;

        if (layerLookup.containsKey(name)) {
            layerLookup.get(name).put(includePhase, layer);
        } else {
            HashMap map = new HashMap();
            map.put(includePhase, layer);
            layerLookup.put(name, map);
        }

        String type = layer.getAttr("type");
        Config config = Config.getInstance();
        if (type.equals("Data") || config.getCustomDataLayers().contains(type)) {
            layer.setKind(Layer.Kind.DATA);
        } else if (type.toLowerCase().endsWith("loss")) {
            layer.setKind(Layer.Kind.LOSS);
        } else {
            layer.setKind(Layer.Kind.INTERMEDIATE);
        }
    }

    public List<Layer> getDataLayers() {
        List<Layer> ret = new ArrayList<>();

        for (Layer layer : layerList) {
            if (layer.getKind() == Layer.Kind.DATA) {
                ret.add(layer);
            }
        }
        return ret;
    }

    public List<Layer> getNonDataLayers() {
        List<Layer> ret = new ArrayList<>();

        for (Layer layer : layerList) {
            if (layer.getKind() != Layer.Kind.DATA) {
                ret.add(layer);
            }
        }
        return ret;
    }

}
