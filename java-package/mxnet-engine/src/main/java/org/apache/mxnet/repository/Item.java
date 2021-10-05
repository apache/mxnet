/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet.repository;

/**
 * {@link Item} is some listed repositories where we can download pre-trained models data. It is
 * used by developers to download specific models data by initialize a {@link Repository}.
 */
public enum Item {
    MLP("mlp", "https://resources.djl.ai/test-models/mlp.tar.gz");

    private String name;
    private String url;

    Item(String name, String url) {
        this.name = name;
        this.url = url;
    }

    /**
     * Gets the name of this {@code Item}.
     *
     * @return the name of this {@code Item}
     */
    public String getName() {
        return name;
    }

    /**
     * Gets the URL of this {@code Item} to download.
     *
     * @return the URL of this {@code Item}
     */
    public String getUrl() {
        return url;
    }
}
