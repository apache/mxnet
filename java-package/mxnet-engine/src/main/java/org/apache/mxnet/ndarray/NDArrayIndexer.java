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

package org.apache.mxnet.ndarray;

import java.util.List;
import java.util.Optional;
import java.util.Stack;
import org.apache.mxnet.engine.OpParams;
import org.apache.mxnet.ndarray.dim.NDIndexBooleans;
import org.apache.mxnet.ndarray.dim.NDIndexElement;
import org.apache.mxnet.ndarray.dim.full.NDIndexFullPick;
import org.apache.mxnet.ndarray.dim.full.NDIndexFullSlice;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.Shape;

/** A helper class for {@link NDArray} implementations for operations with an {@link NDIndex}. */
public class NDArrayIndexer {

    /**
     * Returns a subarray by picking the elements.
     *
     * @param array the array to get from
     * @param index the index to get
     * @return the subArray
     */
    public NDArray get(NDArray array, NDIndex index) {
        if (index.getRank() == 0 && array.getShape().isScalar()) {
            return array.duplicate();
        }

        // use booleanMask for NDIndexBooleans case
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "get() currently didn't support more that one boolean NDArray");
            }
            return array.booleanMask(((NDIndexBooleans) indices.get(0)).getIndex());
        }

        Optional<NDIndexFullPick> fullPick = NDIndexFullPick.fromIndex(index, array.getShape());
        if (fullPick.isPresent()) {
            return get(array, fullPick.get());
        }

        Optional<NDIndexFullSlice> fullSlice = NDIndexFullSlice.fromIndex(index, array.getShape());
        if (fullSlice.isPresent()) {
            return get(array, fullSlice.get());
        }
        throw new UnsupportedOperationException(
                "get() currently supports all, fixed, and slices indices");
    }

    /**
     * Returns a subarray by picking the elements.
     *
     * @param array the array to get from
     * @param fullPick the elements to pick
     * @return the subArray
     */
    public NDArray get(NDArray array, NDIndexFullPick fullPick) {
        OpParams params = new OpParams();
        params.addParam("axis", fullPick.getAxis());
        params.addParam("keepdims", true);
        params.add("mode", "wrap");
        return NDArray.invoke(
                        array.getParent(), "pick", new NDList(array, fullPick.getIndices()), params)
                .singletonOrThrow();
    }

    /**
     * Returns a subarray at the slice.
     *
     * @param array the array to get from
     * @param fullSlice the fullSlice index of the array
     * @return the subArray
     */
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        OpParams params = new OpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        NDArray result = NDArray.invoke(array.getParent(), "_npi_slice", array, params);
        int[] toSqueeze = fullSlice.getToSqueeze();
        if (toSqueeze.length > 0) {
            NDArray oldResult = result;
            result = result.squeeze(toSqueeze);
            oldResult.close();
        }
        return result;
    }

    /**
     * Sets the values of the array at the fullSlice with an array.
     *
     * @param array the array to set
     * @param fullSlice the fullSlice of the index to set in the array
     * @param value the value to set with
     */
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        OpParams params = new OpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        Stack<NDArray> prepareValue = new Stack<>();
        prepareValue.add(value);
        prepareValue.add(prepareValue.peek().toDevice(array.getDevice(), false));
        // prepareValue.add(prepareValue.peek().asType(getDataType(), false));
        // Deal with the case target: (1, 10, 1), original (10)
        // try to find (10, 1) and reshape (10) to that
        Shape targetShape = fullSlice.getShape();
        while (targetShape.size() > value.size()) {
            targetShape = targetShape.slice(1);
        }
        prepareValue.add(prepareValue.peek().reshape(targetShape));
        prepareValue.add(prepareValue.peek().broadcast(fullSlice.getShape()));

        NDArray.invoke(
                "_npi_slice_assign",
                new NDArray[] {array, prepareValue.peek()},
                new NDArray[] {array},
                params);
        for (NDArray toClean : prepareValue) {
            if (toClean != value) {
                toClean.close();
            }
        }
    }

    /**
     * Sets the values of the array at the fullSlice with a number.
     *
     * @param array the array to set
     * @param fullSlice the fullSlice of the index to set in the array
     * @param value the value to set with
     */
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        OpParams params = new OpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());
        params.addParam("scalar", value);
        NDArray.invoke(
                "_npi_slice_assign_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
    }
}
