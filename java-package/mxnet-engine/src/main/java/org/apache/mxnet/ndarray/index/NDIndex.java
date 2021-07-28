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
package org.apache.mxnet.ndarray.index;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.dim.NDIndexAll;
import org.apache.mxnet.ndarray.dim.NDIndexBooleans;
import org.apache.mxnet.ndarray.dim.NDIndexElement;
import org.apache.mxnet.ndarray.dim.NDIndexFixed;
import org.apache.mxnet.ndarray.dim.NDIndexPick;
import org.apache.mxnet.ndarray.dim.NDIndexSlice;
import org.apache.mxnet.ndarray.types.DataType;

/**
 * The {@code NDIndex} allows you to specify a subset of an NDArray that can be used for fetching or
 * updating.
 *
 * <p>It accepts a different index option for each dimension, given in the order of the dimensions.
 * Each dimension has options corresponding to:
 *
 * <ul>
 *   <li>Return all dimensions - Pass null to addIndices
 *   <li>A single value in the dimension - Pass the value to addIndices with a negative index -i
 *       corresponding to [dimensionLength - i]
 *   <li>A range of values - Use addSliceDim
 * </ul>
 *
 * <p>We recommend creating the NDIndex using {@link #NDIndex(String, Object...)}.
 *
 * @see #NDIndex(String, Object...)
 */
public class NDIndex {

    /* Android regex requires escape } char as well */
    private static final Pattern ITEM_PATTERN =
            Pattern.compile(
                    "(\\*)|((-?\\d+|\\{\\})?:(-?\\d+|\\{\\})?(:(-?\\d+|\\{\\}))?)|(-?\\d+|\\{\\})");

    private int rank;
    private List<NDIndexElement> indices;
    private int ellipsisIndex;

    /** Creates an empty {@link NDIndex} to append values to. */
    public NDIndex() {
        rank = 0;
        indices = new ArrayList<>();
        ellipsisIndex = -1;
    }

    /**
     * Creates a {@link NDIndex} given the index values.
     *
     * <p>Here are some examples of the indices format.
     *
     * <pre>
     *     NDArray a = manager.ones(new Shape(5, 4, 3));
     *
     *     // Gets a subsection of the NDArray in the first axis.
     *     assertEquals(a.get(new NDIndex("2")).getShape(), new Shape(4, 3));
     *
     *     // Gets a subsection of the NDArray indexing from the end (-i == length - i).
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(4, 3));
     *
     *     // Gets everything in the first axis and a subsection in the second axis.
     *     // You can use either : or * to represent everything
     *     assertEquals(a.get(new NDIndex(":, 2")).getShape(), new Shape(5, 3));
     *     assertEquals(a.get(new NDIndex("*, 2")).getShape(), new Shape(5, 3));
     *
     *     // Gets a range of values along the second axis that is inclusive on the bottom and exclusive on the top.
     *     assertEquals(a.get(new NDIndex(":, 1:3")).getShape(), new Shape(5, 2, 3));
     *
     *     // Excludes either the min or the max of the range to go all the way to the beginning or end.
     *     assertEquals(a.get(new NDIndex(":, :3")).getShape(), new Shape(5, 3, 3));
     *     assertEquals(a.get(new NDIndex(":, 1:")).getShape(), new Shape(5, 4, 3));
     *
     *     // Uses the value after the second colon in a slicing range, the step, to get every other result.
     *     assertEquals(a.get(new NDIndex(":, 1::2")).getShape(), new Shape(5, 2, 3));
     *
     *     // Uses a negative step to reverse along the dimension.
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(5, 4, 3));
     *
     *     // Uses a variable argument to the index
     *     // It can replace any number in any of these formats with {} and then the value of {}
     *     // is specified in an argument following the indices string.
     *     assertEquals(a.get(new NDIndex("{}, {}:{}", 0, 1, 3)).getShape(), new Shape(2, 3));
     *
     *     // Uses ellipsis to insert many full slices
     *     assertEquals(a.get(new NDIndex("...")).getShape(), new Shape(5, 4, 3));
     *
     *     // Uses ellipsis to select all the dimensions except for last axis where we only get a subsection.
     *     assertEquals(a.get(new NDIndex("..., 2")).getShape(), new Shape(5, 4));
     * </pre>
     *
     * @param indices a comma separated list of indices corresponding to either subsections,
     *     everything, or slices on a particular dimension
     * @param args arguments to replace the variable "{}" in the indices string. Can be an integer,
     *     long, boolean {@link NDArray}, or integer {@link NDArray}.
     * @see <a href="https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html">Numpy
     *     Indexing</a>
     */
    public NDIndex(String indices, Object... args) {
        this();
        addIndices(indices, args);
    }

    /**
     * Creates an NDIndex with the given indices as specified values on the NDArray.
     *
     * @param indices the indices with each index corresponding to the dimensions and negative
     *     indices starting from the end
     */
    public NDIndex(long... indices) {
        this();
        addIndices(indices);
    }

    /**
     * Creates an {@link NDIndex} that just has one slice in the given axis.
     *
     * @param axis the axis to slice
     * @param min the min of the slice
     * @param max the max of the slice
     * @return a new {@link NDIndex} with the given slice.
     */
    public static NDIndex sliceAxis(int axis, long min, long max) {
        NDIndex ind = new NDIndex();
        for (int i = 0; i < axis; i++) {
            ind.addAllDim();
        }
        ind.addSliceDim(min, max);
        return ind;
    }

    /**
     * Returns the number of dimensions specified in the Index.
     *
     * @return the number of dimensions specified in the Index
     */
    public int getRank() {
        return rank;
    }

    /**
     * Returns the index of the ellipsis.
     *
     * @return the index of the ellipsis within this index or -1 for none.
     */
    public int getEllipsisIndex() {
        return ellipsisIndex;
    }

    /**
     * Returns the index affecting the given dimension.
     *
     * @param dimension the affected dimension
     * @return the index affecting the given dimension
     */
    public NDIndexElement get(int dimension) {
        return indices.get(dimension);
    }

    /**
     * Returns the indices.
     *
     * @return the indices
     */
    public List<NDIndexElement> getIndices() {
        return indices;
    }

    /**
     * Updates the NDIndex by appending indices to the array.
     *
     * @param indices the indices to add similar to {@link #NDIndex(String, Object...)}
     * @param args arguments to replace the variable "{}" in the indices string. Can be an integer,
     *     long, boolean {@link NDArray}, or integer {@link NDArray}.
     * @return the updated {@link NDIndex}
     * @see #NDIndex(String, Object...)
     */
    public final NDIndex addIndices(String indices, Object... args) {
        String[] indexItems = indices.split(",");
        rank += indexItems.length;
        int argIndex = 0;
        for (int i = 0; i < indexItems.length; ++i) {
            if ("...".equals(indexItems[i].trim())) {
                // make sure ellipsis appear only once
                if (ellipsisIndex != -1) {
                    throw new IllegalArgumentException(
                            "an index can only have a single ellipsis (\"...\")");
                }
                ellipsisIndex = i;
            } else {
                argIndex = addIndexItem(indexItems[i], argIndex, args);
            }
        }
        if (ellipsisIndex != -1) {
            rank--;
        }
        if (argIndex != args.length) {
            throw new IllegalArgumentException("Incorrect number of index arguments");
        }
        return this;
    }

    /**
     * Updates the NDIndex by appending indices as specified values on the NDArray.
     *
     * @param indices with each index corresponding to the dimensions and negative indices starting
     *     from the end
     * @return the updated {@link NDIndex}
     */
    public final NDIndex addIndices(long... indices) {
        rank += indices.length;
        for (long i : indices) {
            this.indices.add(new NDIndexFixed(i));
        }
        return this;
    }

    /**
     * Updates the NDIndex by appending a boolean NDArray.
     *
     * <p>The NDArray should have a matching shape to the dimensions being fetched and will return
     * where the values in NDIndex do not equal zero.
     *
     * @param index a boolean NDArray where all nonzero elements correspond to elements to return
     * @return the updated {@link NDIndex}
     */
    public NDIndex addBooleanIndex(NDArray index) {
        rank += index.getShape().dimension();
        indices.add(new NDIndexBooleans(index));
        return this;
    }

    /**
     * Appends a new index to get all values in the dimension.
     *
     * @return the updated {@link NDIndex}
     */
    public NDIndex addAllDim() {
        rank++;
        indices.add(new NDIndexAll());
        return this;
    }

    /**
     * Appends multiple new index to get all values in the dimension.
     *
     * @param count how many axes of {@link NDIndexAll} to add.
     * @return the updated {@link NDIndex}
     * @throws IllegalArgumentException if count is negative
     */
    public NDIndex addAllDim(int count) {
        if (count < 0) {
            throw new IllegalArgumentException(
                    "The number of index dimensions to add can't be negative");
        }
        rank += count;
        for (int i = 0; i < count; i++) {
            indices.add(new NDIndexAll());
        }
        return this;
    }

    /**
     * Appends a new index to slice the dimension and returns a range of values.
     *
     * @param min the minimum of the range
     * @param max the maximum of the range
     * @return the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(long min, long max) {
        rank++;
        indices.add(new NDIndexSlice(min, max, null));
        return this;
    }

    /**
     * Appends a new index to slice the dimension and returns a range of values.
     *
     * @param min the minimum of the range
     * @param max the maximum of the range
     * @param step the step of the slice
     * @return the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(long min, long max, long step) {
        rank++;
        indices.add(new NDIndexSlice(min, max, step));
        return this;
    }

    /**
     * Appends a picking index that gets values by index in the axis.
     *
     * @param index the indices should be NDArray. For each element in the indices array, it acts
     *     like a fixed index returning an element of that shape. So, the final shape would be
     *     indices.getShape().addAll(target.getShape().slice(1)) (assuming it is the first index
     *     element).
     * @return the updated {@link NDIndex}
     */
    public NDIndex addPickDim(NDArray index) {
        rank++;
        indices.add(new NDIndexPick(index));
        return this;
    }

    /**
     * Returns a stream of the NDIndexElements.
     *
     * @return a stream of the NDIndexElements
     */
    public Stream<NDIndexElement> stream() {
        return indices.stream();
    }

    private int addIndexItem(String indexItem, int argIndex, Object[] args) {
        indexItem = indexItem.trim();
        Matcher m = ITEM_PATTERN.matcher(indexItem);
        if (!m.matches()) {
            throw new IllegalArgumentException("Invalid argument index: " + indexItem);
        }
        // "*" case
        String star = m.group(1);
        if (star != null) {
            indices.add(new NDIndexAll());
            return argIndex;
        }
        // "number" number only case
        String digit = m.group(7);
        if (digit != null) {
            if ("{}".equals(digit)) {
                Object arg = args[argIndex];
                if (arg instanceof Integer) {
                    indices.add(new NDIndexFixed((Integer) arg));
                    return argIndex + 1;
                } else if (arg instanceof Long) {
                    indices.add(new NDIndexFixed((Long) arg));
                    return argIndex + 1;
                } else if (arg instanceof NDArray) {
                    NDArray array = (NDArray) arg;
                    if (array.getDataType() == DataType.BOOLEAN) {
                        indices.add(new NDIndexBooleans(array));
                        return argIndex + 1;
                    } else if (array.getDataType().isInteger()) {
                        indices.add(new NDIndexPick(array));
                        return argIndex + 1;
                    }
                }
                throw new IllegalArgumentException("Unknown argument: " + arg);
            } else {
                indices.add(new NDIndexFixed(Long.parseLong(digit)));
                return argIndex;
            }
        }

        // Slice
        Long min = null;
        Long max = null;
        Long step = null;
        if (m.group(3) != null) {
            min = parseSliceItem(m.group(3), argIndex, args);
            if ("{}".equals(m.group(3))) {
                argIndex++;
            }
        }
        if (m.group(4) != null) {
            max = parseSliceItem(m.group(4), argIndex, args);
            if ("{}".equals(m.group(4))) {
                argIndex++;
            }
        }
        if (m.group(6) != null) {
            step = parseSliceItem(m.group(6), argIndex, args);
            if ("{}".equals(m.group(6))) {
                argIndex++;
            }
        }
        if (min == null && max == null && step == null) {
            indices.add(new NDIndexAll());
        } else {
            indices.add(new NDIndexSlice(min, max, step));
        }
        return argIndex;
    }

    private Long parseSliceItem(String sliceItem, int argIndex, Object... args) {
        if ("{}".equals(sliceItem)) {
            Object arg = args[argIndex];
            if (arg instanceof Integer) {
                return ((Integer) arg).longValue();
            } else if (arg instanceof Long) {
                return (Long) arg;
            }
            throw new IllegalArgumentException("Unknown slice argument: " + arg);
        } else {
            return Long.parseLong(sliceItem);
        }
    }
}
