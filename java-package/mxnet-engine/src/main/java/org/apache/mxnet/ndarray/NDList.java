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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.types.Shape;

/**
 * An {@code NDList} represents a sequence of {@link NDArray}s with names.
 *
 * <p>Each {@link NDArray} in this list can optionally have a name. You can use the name to look up
 * an NDArray in the NDList.
 *
 * @see NDArray
 */
public class NDList extends ArrayList<NDArray> implements AutoCloseable {
    private static final long serialVersionUID = 1L;

    /** Constructs an empty NDList. */
    public NDList() {}

    /**
     * Constructs an empty NDList with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public NDList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs and initiates an NDList with the specified {@link NDList}s.
     *
     * @param arrays the {@link NDList}s
     */
    public NDList(NDArray... arrays) {
        super(Arrays.asList(arrays));
    }

    /**
     * Constructs and initiates an NDList with the specified {@link NDArray}s.
     *
     * @param other the {@link NDArray}s
     */
    public NDList(Collection<NDArray> other) {
        super(other);
    }

    /**
     * Decodes NDList from byte array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param byteArray byte array to load from
     * @return {@code NDList}
     */
    public static NDList decode(MxResource parent, byte[] byteArray) {
        return decode(parent, new ByteArrayInputStream(byteArray));
    }

    /**
     * Decodes NDList from {@link InputStream}.
     *
     * @param parent {@link MxResource} assigned to {@link NDArray}
     * @param is input stream contains the ndlist information
     * @return {@code NDList}
     */
    public static NDList decode(MxResource parent, InputStream is) {
        try (DataInputStream dis = new DataInputStream(is)) {
            int size = dis.readInt();
            if (size < 0) {
                throw new IllegalArgumentException("Invalid NDList size: " + size);
            }
            NDList list = new NDList();
            for (int i = 0; i < size; i++) {
                list.add(i, NDArray.decode(parent, dis));
            }
            return list;
        } catch (IOException e) {
            throw new IllegalArgumentException("Malformed data", e);
        }
    }

    /**
     * Removes the first occurrence of the specified element from this NDList if it is present.
     *
     * <p>If this list does not contain the element, it is unchanged. More formally, removes the
     * element with the lowest index {@code i} such that {@code
     * (o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))} (if such an element exists).
     *
     * @param name the name of the NDArray to be removed from this NDList, if present
     * @return the element that was removed
     */
    public NDArray remove(String name) {
        int index = 0;
        for (NDArray array : this) {
            if (name.equals(array.getName())) {
                remove(index);
                return array;
            }
            ++index;
        }
        return null;
    }

    /**
     * Returns {@code true} if this NDList contains an NDArray with the specified name.
     *
     * @param name the name of the NDArray to be removed from this NDList, if present
     * @return {@code true} if this list contains the specified element
     */
    public boolean contains(String name) {
        for (NDArray array : this) {
            if (name.equals(array.getName())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns the head index of the NDList.
     *
     * @return the head NDArray
     * @throws IndexOutOfBoundsException if the index is out of range ({@code index &lt; 0 || index
     *     &gt;= size()})
     */
    public NDArray head() {
        return get(0);
    }

    /**
     * Returns the only element if this is a singleton NDList or throws an exception if multiple
     * elements.
     *
     * @return the head NDArray
     * @throws IndexOutOfBoundsException if the list does not contain exactly one element
     */
    public NDArray singletonOrThrow() {
        if (size() != 1) {
            throw new IndexOutOfBoundsException(
                    "Incorrect number of elements in NDList.singletonOrThrow: Expected 1 and was "
                            + size());
        }
        return get(0);
    }

    /**
     * Appends all of the NDArrays in the specified NDList to the end of this NDList, in the order
     * that they are returned by the specified NDList's iterator.
     *
     * @param other the NDList containing NDArray to be added to this list
     * @return this NDList after the addition
     */
    public NDList addAll(NDList other) {
        for (NDArray array : other) {
            add(array);
        }
        return this;
    }

    /**
     * Returns a view of the portion of this NDList between the specified fromIndex, inclusive, and
     * to the end.
     *
     * @param fromIndex the start index (inclusive)
     * @return a view of the portion of this NDList
     */
    public NDList subNDList(int fromIndex) {
        return new NDList(subList(fromIndex, size()));
    }

    /**
     * Converts all the {@code NDArray} in {@code NDList} to a different {@link Device}.
     *
     * @param device the {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the underlying NDArray
     * @return a new {@code NDList} with the NDArrays on specified {@link Device}
     */
    public NDList toDevice(Device device, boolean copy) {
        if (!copy) {
            // if all arrays in NDList are already on device, return itself
            if (this.stream().allMatch(array -> array.getDevice() == device)) {
                return this;
            }
        }
        NDList newNDList = new NDList(size());
        forEach(a -> newNDList.add(a.toDevice(device, copy)));
        return newNDList;
    }

    /**
     * Encodes the NDList to byte array.
     *
     * @return the byte array
     */
    public byte[] encode() {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream dos = new DataOutputStream(baos);
            dos.writeInt(size());
            for (NDArray nd : this) {
                dos.write(nd.encode());
            }
            dos.flush();
            return baos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("NDList is not writable", e);
        }
    }

    /**
     * Gets all of shapes in the {@code NDList}.
     *
     * @return shapes in {@code NDList}
     */
    public Shape[] getShapes() {
        return stream().map(NDArray::getShape).toArray(Shape[]::new);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        forEach(NDArray::close);
        clear();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(200);
        builder.append("NDList size: ").append(size()).append('\n');
        int index = 0;
        for (NDArray array : this) {
            String name = array.getName();
            builder.append(index++).append(' ');
            if (name != null) {
                builder.append(name);
            }
            builder.append(": ")
                    .append(array.getShape())
                    .append(' ')
                    .append(array.getDataType())
                    .append('\n');
        }
        return builder.toString();
    }
}
