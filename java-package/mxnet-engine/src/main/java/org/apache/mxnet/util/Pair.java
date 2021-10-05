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

package org.apache.mxnet.util;

import java.util.Objects;

/**
 * A class containing the key-value pair.
 *
 * @param <K> the key type
 * @param <V> the value type
 */
public class Pair<K, V> {

    private K key;
    private V value;

    /**
     * Constructs a {@code Pair} instance with key and value.
     *
     * @param key the key
     * @param value the value
     */
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    /**
     * Returns the key of this pair.
     *
     * @return the key
     */
    public K getKey() {
        return key;
    }

    /**
     * Returns the value of this pair.
     *
     * @return the value
     */
    public V getValue() {
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Pair<?, ?> pair = (Pair<?, ?>) o;
        return Objects.equals(key, pair.key) && value.equals(pair.value);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(key, value);
    }
}
