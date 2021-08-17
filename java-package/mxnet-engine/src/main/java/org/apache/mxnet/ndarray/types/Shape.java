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

package org.apache.mxnet.ndarray.types;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

/** A class that presents the {@link NDArray}'s shape information. */
public class Shape {

    private long[] shape;
    private LayoutType[] layout;

    /**
     * Constructs and initializes a {@code Shape} with specified dimension as {@code (long...
     * shape)}.
     *
     * @param shape the dimensions of the shape
     * @throws IllegalArgumentException Thrown if any element in Shape is invalid. It should not be
     *     less than -1. Also thrown if the shape and layout do not have equal sizes.
     */
    public Shape(long... shape) {
        this(
                shape,
                Arrays.stream(shape).mapToObj(x -> LayoutType.UNKNOWN).toArray(LayoutType[]::new));
    }

    /**
     * Constructs and initializes a {@code Shape} with specified dimension.
     *
     * @param shape the dimensions of the shape
     * @throws IllegalArgumentException Thrown if any element in Shape is invalid. It should not be
     *     less than -1. Also thrown if the shape and layout do not have equal sizes.
     */
    public Shape(List<Long> shape) {
        this(
                shape.stream().mapToLong(l -> l).toArray(),
                shape.stream().map(x -> LayoutType.UNKNOWN).toArray(LayoutType[]::new));
    }

    /**
     * Constructs and initializes a {@code Shape} with specified shape and layout pairList.
     *
     * @param shape the dimensions and layout of the shape
     * @throws IllegalArgumentException Thrown if any element in Shape is invalid. It should not be
     *     less than -1 .Also thrown if the shape and layout do not have equal sizes.
     */
    public Shape(PairList<Long, LayoutType> shape) {
        this(
                shape.keys().stream().mapToLong(l -> l).toArray(),
                shape.values().toArray(new LayoutType[shape.size()]));
    }

    /**
     * Constructs and initializes a {@code Shape} with specified dimension and layout.
     *
     * @param shape the size of each axis of the shape
     * @param layout the {@link LayoutType} of each axis in the shape
     * @throws IllegalArgumentException Thrown if any element in Shape is invalid. It should not be
     *     less than -1. Also thrown for an invalid layout. Also thrown if the shape and layout do
     *     not have equal sizes.
     */
    public Shape(long[] shape, String layout) {
        this(shape, LayoutType.fromValue(layout));
    }

    /**
     * Constructs and initializes a {@code Shape} with specified dimension and layout.
     *
     * @param shape the size of each axis of the shape
     * @param layout the {@link LayoutType} of each axis in the shape
     * @throws IllegalArgumentException Thrown if any element in Shape is invalid. It should not be
     *     less than -1. Also thrown if the shape and layout do not have equal sizes.
     */
    public Shape(long[] shape, LayoutType[] layout) {
        if (Arrays.stream(shape).anyMatch(s -> s < -1)) {
            throw new IllegalArgumentException("The shape must be >= -1");
        }
        if (shape.length != layout.length) {
            throw new IllegalArgumentException("The shape and layout must have the same length");
        }
        this.shape = shape;
        this.layout = layout;
    }

    /**
     * Returns a new shape altering the given dimension.
     *
     * @param shape the shape to update
     * @param dimension the dimension to get the shape in
     * @param value the value to set the dimension to
     * @return a new shape with the update applied
     */
    public static Shape update(Shape shape, int dimension, long value) {
        long[] newShape = shape.shape.clone();
        newShape[dimension] = value;
        return new Shape(newShape, shape.layout);
    }

    /**
     * Returns the dimensions of the {@code Shape}.
     *
     * @return the dimensions of the {@code Shape}
     */
    public long[] getShape() {
        return shape;
    }

    /**
     * Returns the shape in the given dimension.
     *
     * @param dimension the dimension to get the shape in
     * @return the shape in the given dimension
     */
    public long get(int dimension) {
        return shape[dimension];
    }

    /**
     * Returns the layout type in the given dimension.
     *
     * @param dimension the dimension to get the layout type in
     * @return the layout type in the given dimension
     */
    public LayoutType getLayoutType(int dimension) {
        return layout[dimension];
    }

    /**
     * Returns the size of a specific dimension or several specific dimensions.
     *
     * @param dimensions the dimension or dimensions to find the size of
     * @return the size of specific dimension(s) or -1 for indeterminate size
     * @throws IllegalArgumentException thrown if passed an invalid dimension
     */
    public long size(int... dimensions) {
        long total = 1;
        for (long d : dimensions) {
            if (d < 0 || d >= shape.length) {
                throw new IllegalArgumentException("Invalid dimension " + d);
            }
            if (shape[Math.toIntExact(d)] == -1) {
                return -1;
            }
            total *= shape[Math.toIntExact(d)];
        }
        return total;
    }

    /**
     * Returns the total size.
     *
     * @return the total size or -1 for indeterminate size
     */
    public long size() {
        long total = 1;
        for (long v : shape) {
            if (v == -1) {
                return -1;
            }
            total *= v;
        }
        return total;
    }

    /**
     * Returns the number of dimensions of this {@code Shape}.
     *
     * @return the number of dimensions of this {@code Shape}
     */
    public int dimension() {
        return shape.length;
    }

    /**
     * Return the count of unknown value in this {@code Shape}.
     *
     * @return the number of unknown value in this {@code Shape}
     */
    public long getUnknownValueCount() {
        return Arrays.stream(shape).filter(s -> s == -1).count();
    }

    /**
     * Creates a new {@code Shape} whose content is a slice of this shape.
     *
     * <p>The sub shape begins at the specified {@code beginIndex} and extends to {@code endIndex -
     * 1}.
     *
     * @param beginIndex the beginning index, inclusive
     * @return a new {@code Shape} whose content is a slice of this shape
     */
    public Shape slice(int beginIndex) {
        return slice(beginIndex, shape.length);
    }

    /**
     * Creates a new {@code Shape} whose content is a slice of this shape.
     *
     * <p>The sub shape begins at the specified {@code beginIndex} and extends to {@code endIndex -
     * 1}.
     *
     * @param beginIndex the beginning index, inclusive
     * @param endIndex the ending index, exclusive
     * @return a new {@code Shape} whose content is a slice of this shape
     */
    public Shape slice(int beginIndex, int endIndex) {
        int size = endIndex - beginIndex;
        long[] out = new long[size];
        System.arraycopy(shape, beginIndex, out, 0, size);
        return new Shape(out);
    }

    /**
     * Returns only the axes of the Shape whose layout types match the predicate.
     *
     * @param predicate the predicate to compare the axes of the Shape with
     * @return a new filtered Shape
     */
    public Shape filterByLayoutType(Predicate<LayoutType> predicate) {
        return new Shape(
                new PairList<>(
                        this.stream()
                                .filter(pair -> predicate.test(pair.getValue()))
                                .collect(Collectors.toList())));
    }

    /**
     * Returns a mapped shape.
     *
     * @param mapper the function to map each element of the Shape by
     * @return a new mapped Shape
     */
    public Shape map(Function<Pair<Long, LayoutType>, Pair<Long, LayoutType>> mapper) {
        return new Shape(new PairList<>(stream().map(mapper).collect(Collectors.toList())));
    }

    /**
     * Returns a stream of the Shape.
     *
     * @return the stream of the Shape
     */
    public Stream<Pair<Long, LayoutType>> stream() {
        return new PairList<>(
                        Arrays.stream(shape).boxed().collect(Collectors.toList()),
                        Arrays.asList(layout))
                .stream();
    }

    /**
     * Joins this shape with axes.
     *
     * @param axes the axes to join
     * @return the joined {@code Shape}
     */
    public Shape add(long... axes) {
        return this.addAll(new Shape(axes));
    }

    /**
     * Joins this shape with specified {@code other} shape.
     *
     * @param other the shape to join
     * @return the joined {@code Shape}
     */
    public Shape addAll(Shape other) {
        return new Shape(
                LongStream.concat(Arrays.stream(shape), Arrays.stream(other.shape)).toArray());
    }

    /**
     * Returns the head index of the shape.
     *
     * @return the head index of the shape
     * @throws IndexOutOfBoundsException Thrown if the shape is empty
     */
    public long head() {
        // scalar case
        if (shape.length == 0) {
            throw new IndexOutOfBoundsException("can't get value from scalar shape.");
        }
        return shape[0];
    }

    /**
     * Returns the tail index of the shape.
     *
     * @return the tail index of the shape
     * @throws IndexOutOfBoundsException Thrown if the shape is empty
     */
    public long tail() {
        // scalar case
        if (shape.length == 0) {
            throw new IndexOutOfBoundsException("can't get value from scalar shape.");
        }
        return shape[shape.length - 1];
    }

    /**
     * Returns the number of trailing ones in the array shape.
     *
     * <p>For example, a rank 3 array with shape [10, 1, 1] would return 2 for this method
     *
     * @return the number of trailing ones in the shape
     */
    public int getTrailingOnes() {
        for (int i = 0; i < shape.length; i++) {
            if (shape[shape.length - i - 1] != 1) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Returns the number of leading ones in the array shape.
     *
     * <p>For example, a rank 3 array with shape [1, 10, 1] would return value 1 for this method
     *
     * @return the number of leading ones in the shape
     */
    public int getLeadingOnes() {
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Returns {@code true} if the NDArray is a scalar.
     *
     * @return whether the NDArray is a scalar
     */
    public boolean isScalar() {
        return dimension() == 0;
    }

    /**
     * Returns {@code true} if the NDArray contains zero dimensions.
     *
     * @return whether the NDArray contain zero dimensions
     */
    public boolean hasZeroDimension() {
        for (int i = 0; i < dimension(); i++) {
            if (shape[i] == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns {@code true} if a layout is set.
     *
     * @return whether a layout has been set
     */
    public boolean isLayoutKnown() {
        return !Arrays.stream(layout).allMatch(l -> l == LayoutType.UNKNOWN);
    }

    /**
     * Returns the layout type for each axis in this shape.
     *
     * @return the layout type for each axis in this shape
     */
    public LayoutType[] getLayout() {
        return layout;
    }

    /**
     * Returns the string layout type for each axis in this shape.
     *
     * @return the string layout type for each axis in this shape
     */
    public String toLayoutString() {
        return LayoutType.toString(layout);
    }

    /**
     * Gets the byte array representation of this {@code Shape} for serialization.
     *
     * @return a byte array representation of this {@code Shape}
     */
    public byte[] getEncoded() {
        int length = 8 + shape.length * 8 + layout.length * 2;
        ByteBuffer bb = ByteBuffer.allocate(length);
        bb.putInt(shape.length);
        for (long l : shape) {
            bb.putLong(l);
        }
        bb.putInt(layout.length);
        for (LayoutType layoutType : layout) {
            bb.putChar(layoutType.getValue());
        }
        return bb.array();
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
        Shape shape1 = (Shape) o;
        return Arrays.equals(shape, shape1.shape);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Arrays.hashCode(shape);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < shape.length; ++i) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(shape[i]);
        }
        sb.append(')');
        return sb.toString();
    }

    /**
     * Decodes the data in the given {@link DataInputStream} and converts it into the corresponding
     * {@link Shape} object.
     *
     * @param dis the inputstream to read from
     * @return the corresponding {@link Shape} object
     * @throws IOException when an I/O error occurs
     */
    public static Shape decode(DataInputStream dis) throws IOException {
        // Shape
        int length = dis.readInt();
        long[] shapeValue = new long[length];
        for (int i = 0; i < length; ++i) {
            shapeValue[i] = dis.readLong();
        }

        // Layout
        length = dis.readInt();
        char[] layout = new char[length];
        for (int i = 0; i < length; ++i) {
            layout[i] = dis.readChar();
        }
        return new Shape(shapeValue, new String(layout));
    }
}
