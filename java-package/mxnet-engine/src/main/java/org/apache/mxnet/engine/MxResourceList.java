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

package org.apache.mxnet.engine;

import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

import java.util.List;
import java.util.Map;

public class MxResourceList extends PairList<String, MxResource> {

    /** Creates an empty {@code MxResourceList}. */
    public MxResourceList() {}

    /**
     * Constructs an empty {@code MxResourceList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public MxResourceList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code MxResourceList}
     * @param values the value list containing the elements to be placed into this {@code MxResource}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public MxResourceList(List<String> keys, List<MxResource> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code MxResourceList}
     */
    public MxResourceList(List<Pair<String, MxResource>> list) {
        super(list);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public MxResourceList(Map<String, MxResource> map) {
        super(map);
    }

}
