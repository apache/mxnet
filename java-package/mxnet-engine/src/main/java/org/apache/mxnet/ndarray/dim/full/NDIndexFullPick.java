/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.apache.mxnet.ndarray.dim.full;

import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.dim.NDIndexAll;
import org.apache.mxnet.ndarray.dim.NDIndexElement;
import org.apache.mxnet.ndarray.dim.NDIndexPick;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.Shape;

import java.util.Optional;

/** A simplified representation of a pick-based {@link MxNDArray}. */
public final class NDIndexFullPick {

    private MxNDArray indices;
    private int axis;

    /**
     * Constructs a new {@link NDIndexFullPick}.
     *
     * @param indices the indices to pick
     * @param axis the axis to pick at
     */
    private NDIndexFullPick(MxNDArray indices, int axis) {
        this.indices = indices;
        this.axis = axis;
    }

    /**
     * Returns (if possible) the {@link NDIndexFullPick} representation of an {@link NDIndex}.
     *
     * @param index the index to represent
     * @param target the shape of the array to index
     * @return the full pick representation or nothing if it can't represent the index
     */
    public static Optional<NDIndexFullPick> fromIndex(NDIndex index, Shape target) {
        int axis = 0;
        NDIndexFullPick fullPick = null;
        for (NDIndexElement el : index.getIndices()) {
            if (el instanceof NDIndexAll) {
                axis++;
            } else if (el instanceof NDIndexPick) {
                if (fullPick == null) {
                    fullPick = new NDIndexFullPick(((NDIndexPick) el).getIndices(), axis);
                } else {
                    // Don't support multiple picks
                    throw new UnsupportedOperationException(
                            "Only one pick per get is currently supported");
                }
            } else {
                // Invalid dim for fullPick
                return Optional.empty();
            }
        }
        return Optional.ofNullable(fullPick);
    }

    /**
     * Returns the indices to pick.
     *
     * @return the indices to pick
     */
    public MxNDArray getIndices() {
        return indices;
    }

    /**
     * Returns the axis to pick.
     *
     * @return the axis to pick
     */
    public int getAxis() {
        return axis;
    }
}
