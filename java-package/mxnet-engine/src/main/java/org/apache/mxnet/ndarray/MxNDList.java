package org.apache.mxnet.ndarray;

import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.util.MxNDArrayUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class MxNDList extends ArrayList<MxNDArray> implements AutoCloseable {
    private static final long serialVersionUID = 1L;

    /** Constructs an empty NDList. */
    public MxNDList() {}

    /**
     * Constructs an empty NDList with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public MxNDList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs and initiates an NDList with the specified {@link MxNDList}s.
     *
     * @param arrays the {@link MxNDList}s
     */
    public MxNDList(MxNDArray... arrays) {
        super(Arrays.asList(arrays));
    }

    /**
     * Constructs and initiates an NDList with the specified {@link MxNDArray}s.
     *
     * @param other the {@link MxNDArray}s
     */
    public MxNDList(Collection<MxNDArray> other) {
        super(other);
    }

    /**
     * Decodes NDList from byte array.
     *
     * @param byteArray byte array to load from
     * @return {@code NDList}
     */
    public static MxNDList decode(MxResource parent, byte[] byteArray) {
        return decode(parent, new ByteArrayInputStream(byteArray));
    }

    /**
     * Decodes NDList from {@link InputStream}.
     *
     * @param parent manager assigned to {@link MxNDArray}
     * @param is input stream contains the ndlist information
     * @return {@code NDList}
     */
    public static MxNDList decode(MxResource parent, InputStream is) {
        try (DataInputStream dis = new DataInputStream(is)) {
            int size = dis.readInt();
            if (size < 0) {
                throw new IllegalArgumentException("Invalid NDList size: " + size);
            }
            MxNDList list = new MxNDList();
            for (int i = 0; i < size; i++) {
                list.add(i, MxNDArrayUtils.decode(parent, dis));
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
    public MxNDArray remove(String name) {
        int index = 0;
        for (MxNDArray array : this) {
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
        for (MxNDArray array : this) {
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
    public MxNDArray head() {
        return get(0);
    }

    /**
     * Returns the only element if this is a singleton NDList or throws an exception if multiple
     * elements.
     *
     * @return the head NDArray
     * @throws IndexOutOfBoundsException if the list does not contain exactly one element
     */
    public MxNDArray singletonOrThrow() {
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
    public MxNDList addAll(MxNDList other) {
        for (MxNDArray array : other) {
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
    public MxNDList subNDList(int fromIndex) {
        return new MxNDList(subList(fromIndex, size()));
    }

    /**
     * Converts all the {@code NDArray} in {@code NDList} to a different {@link Device}.
     *
     * @param device the {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the underlying NDArray
     * @return a new {@code NDList} with the NDArrays on specified {@link Device}
     */
    public MxNDList toDevice(Device device, boolean copy) {
        if (!copy) {
            // if all arrays in NDList are already on device, return itself
            if (this.stream().allMatch(array -> array.getDevice() == device)) {
                return this;
            }
        }
        MxNDList newNDList = new MxNDList(size());
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
            for (MxNDArray nd : this) {
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
        return stream().map(MxNDArray::getShape).toArray(Shape[]::new);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        forEach(MxNDArray::close);
        clear();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(200);
        builder.append("NDList size: ").append(size()).append('\n');
        int index = 0;
        for (MxNDArray array : this) {
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
