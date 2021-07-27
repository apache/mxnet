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

package org.apache.mxnet.translate;

import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** {@code Pipeline} allows applying multiple transforms on an input {@link NDList}. */
public class Pipeline {

    private PairList<IndexKey, Transform> transforms;

    /** Creates a new instance of {@code Pipeline} that has no {@link Transform} defined yet. */
    public Pipeline() {
        transforms = new PairList<>();
    }

    /**
     * Creates a new instance of {@code Pipeline} that can apply the given transforms on its input.
     *
     * <p>Since no keys are provided for these transforms, they will be applied to the first element
     * in the input {@link NDList} when the {@link #transform(NDList) transform} method is called on
     * this object.
     *
     * @param transforms the transforms to be applied when the {@link #transform(NDList) transform}
     *     method is called on this object
     */
    public Pipeline(Transform... transforms) {
        this.transforms = new PairList<>();
        for (Transform transform : transforms) {
            this.transforms.add(new IndexKey(0), transform);
        }
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the input when
     * the {@link #transform(NDList) transform} method is called on this object.
     *
     * <p>Since no keys are provided for this {@link Transform}, it will be applied to the first
     * element in the input {@link NDList}.
     *
     * @param transform the {@link Transform} to be added
     * @return this {@code Pipeline}
     */
    public Pipeline add(Transform transform) {
        transforms.add(new IndexKey(0), transform);
        return this;
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the {@link
     * NDArray} at the given index in the input {@link NDList}.
     *
     * @param index the index corresponding to the {@link NDArray} in the input {@link NDList} on
     *     which the given transform must be applied to
     * @param transform the {@link Transform} to be added
     * @return this {@code Pipeline}
     */
    public Pipeline add(int index, Transform transform) {
        transforms.add(new IndexKey(index), transform);
        return this;
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the {@link
     * NDArray} with the given key as name in the input {@link NDList}.
     *
     * @param name the key corresponding to the {@link NDArray} in the input {@link NDList} on which
     *     the given transform must be applied to
     * @param transform the {@code Transform} to be applied when the {@link #transform(NDList)
     *     transform} method is called on this object
     * @return this {@code Pipeline}
     */
    public Pipeline add(String name, Transform transform) {
        transforms.add(new IndexKey(name), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position.
     *
     * <p>Since no keys or indices are provided for this {@link Transform}, it will be applied to
     * the first element in the input {@link NDList} when the {@link #transform(NDList) transform}
     * method is called on this object.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, Transform transform) {
        transforms.add(position, new IndexKey(0), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position to be
     * applied on the {@link NDArray} at the given index in the input {@link NDList}.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param index the index corresponding to the {@link NDArray} in the input {@link NDList} on
     *     which the given transform must be applied to
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, int index, Transform transform) {
        transforms.add(position, new IndexKey(index), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position to be
     * applied on the {@link NDArray} with the given name in the input {@link NDList}.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param name the key corresponding to the {@link NDArray} in the input {@link NDList} on which
     *     the given transform must be applied to
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, String name, Transform transform) {
        transforms.add(position, new IndexKey(name), transform);
        return this;
    }

    /**
     * Applies the transforms configured in this object on the input {@link NDList}.
     *
     * <p>If a key is specified with the transform, those transforms will only be applied to the
     * {@link NDArray} in the input {@link NDList}. If a key is not specified, it will be applied to
     * the first element in the input {@link NDList}.
     *
     * @param input the input {@link NDList} on which the tranforms are to be applied
     * @return the output {@link NDList} after applying the tranforms
     */
    public NDList transform(NDList input) {
        if (transforms.isEmpty() || input.isEmpty()) {
            return input;
        }

        NDArray[] arrays = input.toArray(new NDArray[0]);

        Map<IndexKey, Integer> map = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.get(i).getName();
            if (key != null) {
                map.put(new IndexKey(key), i);
            }
            map.put(new IndexKey(i), i);
        }
        // apply transform
        for (Pair<IndexKey, Transform> transform : transforms) {
            IndexKey key = transform.getKey();
            int index = map.get(key);
            NDArray array = arrays[index];

            arrays[index] = transform.getValue().transform(array);
            arrays[index].setName(array.getName());
        }

        return new NDList(arrays);
    }

    private static final class IndexKey {
        private String key;
        private int index;

        private IndexKey(String key) {
            this.key = key;
        }

        private IndexKey(int index) {
            this.index = index;
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            if (key == null) {
                return index;
            }
            return key.hashCode();
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof IndexKey)) {
                return false;
            }
            IndexKey other = (IndexKey) obj;
            if (key == null) {
                return index == other.index;
            }
            return key.equals(other.key);
        }
    }
}
