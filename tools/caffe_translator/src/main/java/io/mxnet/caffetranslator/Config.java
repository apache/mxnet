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
 * \file Config.java
 * \brief Helper class to store config
 */

package io.mxnet.caffetranslator;

import java.util.List;
import java.util.Vector;

public class Config {

    private static final Config instance = new Config();

    public static Config getInstance() {
        return instance;
    }

    private Config() {
        if (instance != null) {
            throw new IllegalStateException("Already instantiated");
        }

        customDataLayers = new Vector<String>();
    }

    public List<String> getCustomDataLayers() {
        return customDataLayers;
    }

    public void addCustomDataLayer(String name) {
        customDataLayers.add(name);
    }

    private Vector<String> customDataLayers;
}
