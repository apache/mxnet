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
 * \file Layer.java
 * \brief Model for a layer
 */

package io.mxnet.caffetranslator;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Layer {

    @Getter
    @Setter
    private String name;

    @Getter
    @Setter
    private int layerIndex;

    @Getter
    @Setter
    private Kind kind;

    @Getter
    @Setter
    private String prototxt;

    @Getter
    private final List<String> bottoms;

    @Getter
    private final List<String> tops;

    @Setter
    @Getter
    private List<Map<String, String>> params;

    @Setter
    private Map<String, List<String>> attr;

    public Layer() {
        tops = new ArrayList<>();
        bottoms = new ArrayList<>();
        attr = new HashMap<>();
        params = new ArrayList<>();
    }

    public Layer(int layerIndex) {
        this();
        this.layerIndex = layerIndex;
    }

    public void addAttr(String key, String value) {
        List<String> list = attr.get(key);
        if (list == null) {
            list = new ArrayList<String>();
            list.add(value);
            attr.put(key, list);
        } else {
            list.add(value);
        }
    }

    public String getAttr(String key) {
        List<String> list = attr.get(key);
        if (list == null) {
            return null;
        }

        return list.get(0);
    }

    public String getAttr(String key, String defaultValue) {
        String attr = getAttr(key);
        return attr != null ? attr : defaultValue;
    }

    public boolean hasAttr(String key) {
        return attr.containsKey(key);
    }

    public boolean attrEquals(String key, String value) {
        if (!attr.containsKey(key)) {
            return false;
        }
        return getAttr(key).equals(value);
    }

    public List<String> getAttrList(String key) {
        return attr.get(key);
    }

    public void addTop(String top) {
        tops.add(top);
    }

    public void addBottom(String bottom) {
        bottoms.add(bottom);
    }

    public String getBottom() {
        return bottoms.size() > 0 ? bottoms.get(0) : null;
    }

    public String getType() {
        return attr.get("type").get(0);
    }

    public String getTop() {
        return tops.size() > 0 ? tops.get(0) : null;
    }

    public enum Kind {
        DATA, INTERMEDIATE, LOSS;
    }
}
