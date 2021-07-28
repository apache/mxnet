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

package org.apache.mxnet.nn;

import java.util.List;
import java.util.Map;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

/** Represents a set of names and Parameters. */
public class ParameterList extends PairList<String, Parameter> {

    /** Create an empty {@code ParameterList}. */
    public ParameterList() {}

    /**
     * Constructs an empty {@code ParameterList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public ParameterList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code ParameterList}
     * @param values the value list containing the elements to be placed into this {@code
     *     ParameterList}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public ParameterList(List<String> keys, List<Parameter> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code ParameterList}
     */
    public ParameterList(List<Pair<String, Parameter>> list) {
        super(list);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public ParameterList(Map<String, Parameter> map) {
        super(map);
    }
}
