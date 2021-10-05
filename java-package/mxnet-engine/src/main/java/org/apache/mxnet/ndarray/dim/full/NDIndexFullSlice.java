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
package org.apache.mxnet.ndarray.dim.full;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.apache.mxnet.ndarray.dim.NDIndexAll;
import org.apache.mxnet.ndarray.dim.NDIndexElement;
import org.apache.mxnet.ndarray.dim.NDIndexFixed;
import org.apache.mxnet.ndarray.dim.NDIndexSlice;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.Shape;

/** An index as a slice on all dimensions where some dimensions can be squeezed. */
public final class NDIndexFullSlice {
    private long[] min;
    private long[] max;
    private long[] step;
    private int[] toSqueeze;
    private Shape shape;
    private Shape squeezedShape;

    /**
     * Constructs a {@link NDIndexFullSlice}.
     *
     * @param min the min for each axis
     * @param max the max for each axis
     * @param step the step for each axis
     * @param toSqueeze the axes to squeeze after slicing
     * @param shape the result shape (without squeezing)
     * @param squeezedShape the result shape (with squeezing)
     */
    private NDIndexFullSlice(
            long[] min,
            long[] max,
            long[] step,
            int[] toSqueeze,
            Shape shape,
            Shape squeezedShape) {
        this.min = min;
        this.max = max;
        this.step = step;
        this.toSqueeze = toSqueeze;
        this.shape = shape;
        this.squeezedShape = squeezedShape;
    }

    /**
     * Returns (if possible) the {@link NDIndexFullSlice} representation of an {@link NDIndex}.
     *
     * @param index the index to represent
     * @param target the shape of the array to index
     * @return the full slice representation or nothing if it can't represent the index
     */
    public static Optional<NDIndexFullSlice> fromIndex(NDIndex index, Shape target) {
        if (!index.stream()
                .allMatch(
                        ie ->
                                ie instanceof NDIndexAll
                                        || ie instanceof NDIndexFixed
                                        || ie instanceof NDIndexSlice)) {
            return Optional.empty();
        }
        int ellipsisIndex = index.getEllipsisIndex();
        int indDimensions = index.getRank();
        int targetDimensions = target.dimension();
        if (indDimensions > target.dimension()) {
            throw new IllegalArgumentException(
                    "The index has too many dimensions - "
                            + indDimensions
                            + " dimensions for array with "
                            + targetDimensions
                            + " dimensions");
        }
        long[] min = new long[targetDimensions];
        long[] max = new long[targetDimensions];
        long[] step = new long[targetDimensions];
        List<Integer> toSqueeze = new ArrayList<>(targetDimensions);
        long[] shape = new long[targetDimensions];
        List<Long> squeezedShape = new ArrayList<>(targetDimensions);
        if (ellipsisIndex == -1 || ellipsisIndex == indDimensions) {
            // ellipsis in the end and non ellipsis case
            for (int i = 0; i < indDimensions; i++) {
                NDIndexElement ie = index.get(i);
                addSliceInfo(ie, i, target, min, max, step, toSqueeze, shape, squeezedShape);
            }
            for (int i = indDimensions; i < target.dimension(); i++) {
                padIndexAll(i, target, min, max, step, shape, squeezedShape);
            }
        } else if (ellipsisIndex == 0) {
            // ellipsis in the beginning
            int paddingDim = targetDimensions - indDimensions;
            int i;
            for (i = 0; i < paddingDim; ++i) {
                padIndexAll(i, target, min, max, step, shape, squeezedShape);
            }
            for (; i < targetDimensions; ++i) {
                NDIndexElement ie = index.get(i - paddingDim);
                addSliceInfo(ie, i, target, min, max, step, toSqueeze, shape, squeezedShape);
            }
        } else {
            // ellipsis in the middle
            int paddingDim = targetDimensions - indDimensions;
            int i;
            for (i = 0; i < ellipsisIndex; ++i) {
                NDIndexElement ie = index.get(i);
                addSliceInfo(ie, i, target, min, max, step, toSqueeze, shape, squeezedShape);
            }
            for (; i < paddingDim + ellipsisIndex; ++i) {
                padIndexAll(i, target, min, max, step, shape, squeezedShape);
            }
            for (; i < targetDimensions; ++i) {
                NDIndexElement ie = index.get(i - paddingDim);
                addSliceInfo(ie, i, target, min, max, step, toSqueeze, shape, squeezedShape);
            }
        }
        int[] squeeze = toSqueeze.stream().mapToInt(i -> i).toArray();
        NDIndexFullSlice fullSlice =
                new NDIndexFullSlice(
                        min, max, step, squeeze, new Shape(shape), new Shape(squeezedShape));
        return Optional.of(fullSlice);
    }

    private static void addSliceInfo(
            NDIndexElement ie,
            int i,
            Shape target,
            long[] min,
            long[] max,
            long[] step,
            List<Integer> toSqueeze,
            long[] shape,
            List<Long> squeezedShape) {
        if (ie instanceof NDIndexFixed) {
            NDIndexFixed fixed = ((NDIndexFixed) ie);
            long rawIndex = fixed.getIndex();
            min[i] = rawIndex < 0 ? Math.floorMod(rawIndex, target.get(i)) : rawIndex;
            max[i] = min[i] + 1;
            step[i] = 1;
            toSqueeze.add(i);
            shape[i] = 1;
        } else if (ie instanceof NDIndexSlice) {
            NDIndexSlice slice = (NDIndexSlice) ie;
            long rawMin = Optional.ofNullable(slice.getMin()).orElse(0L);
            min[i] = rawMin < 0 ? Math.floorMod(rawMin, target.get(i)) : rawMin;
            long rawMax = Optional.ofNullable(slice.getMax()).orElse(target.size(i));
            max[i] = rawMax < 0 ? Math.floorMod(rawMax, target.get(i)) : rawMax;
            step[i] = Optional.ofNullable(slice.getStep()).orElse(1L);
            shape[i] = (long) Math.ceil(((double) (max[i] - min[i])) / step[i]);
            squeezedShape.add(shape[i]);
        } else if (ie instanceof NDIndexAll) {
            padIndexAll(i, target, min, max, step, shape, squeezedShape);
        }
    }

    private static void padIndexAll(
            int i,
            Shape target,
            long[] min,
            long[] max,
            long[] step,
            long[] shape,
            List<Long> squeezedShape) {
        min[i] = 0;
        max[i] = target.size(i);
        step[i] = 1;
        shape[i] = target.size(i);
        squeezedShape.add(target.size(i));
    }

    /**
     * Returns the slice min for each axis.
     *
     * @return the slice min for each axis
     */
    public long[] getMin() {
        return min;
    }

    /**
     * Returns the slice max for each axis.
     *
     * @return the slice max for each axis
     */
    public long[] getMax() {
        return max;
    }

    /**
     * Returns the slice step for each axis.
     *
     * @return the slice step for each axis
     */
    public long[] getStep() {
        return step;
    }

    /**
     * Returns the squeeze array of axis.
     *
     * @return the squeeze array of axis
     */
    public int[] getToSqueeze() {
        return toSqueeze;
    }

    /**
     * Returns the slice shape without squeezing.
     *
     * @return the slice shape without squeezing
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Returns the slice shape with squeezing.
     *
     * @return the slice shape with squeezing
     */
    public Shape getSqueezedShape() {
        return squeezedShape;
    }
}
