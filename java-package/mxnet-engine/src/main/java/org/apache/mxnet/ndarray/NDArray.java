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

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;
import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.GradReq;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.OpParams;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;
import org.apache.mxnet.util.Float16Utils;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Class representing an n-dimensional array.
 *
 * <p>NDArray is the core data structure for all mathematical computations. An NDArray represents a
 * multidimensional, fixed-size homogeneous array. It has very similar behaviour to the Numpy python
 * package with the addition of efficient computing.
 */
public class NDArray extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(NDArray.class);

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;
    private static final NDArray[] EMPTY = new NDArray[0];

    private String name;
    private Device device;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    // use Boolean object to maintain three status: false, true
    // and null which means the flag is not set by the native engine yet
    private Boolean hasGradient;
    private Integer version;
    private NDArrayEx mxNDArrayEx;

    protected NDArray(Pointer handle) {
        super(BaseMxResource.getSystemMxResource(), handle);
    }

    /**
     * Constructs an {@link NDArray} from a native handle and metadata (internal. Use {@method
     * create} methods).
     *
     * @param parent the parent {@link MxResource} to manage the life circle of the {@link NDArray}
     * @param handle the pointer to the native NDArray memory
     * @param device the device the new array will be located on
     * @param shape the shape of the new array
     * @param dataType the dataType of the new array
     * @param hasGradient the gradient status of the new array
     */
    NDArray(
            MxResource parent,
            Pointer handle,
            Device device,
            Shape shape,
            DataType dataType,
            boolean hasGradient) {
        this(parent, handle);
        setParent(parent);
        this.device = device;
        // shape check
        if (Arrays.stream(shape.getShape()).anyMatch(s -> s < 0)) {
            throw new IllegalArgumentException("The shape must be >= 0");
        }
        this.shape = shape;
        this.dataType = dataType;
        this.hasGradient = hasGradient;
        if (parent != null) {
            parent.addSubResource(this);
        }
    }

    /**
     * Constructs an {@link NDArray} from a native handle and metadata (internal).
     *
     * @param parent the parent {@link MxResource} to manage the life circle of the {@link NDArray}
     * @param handle the pointer to the native NDArray memory
     */
    NDArray(MxResource parent, Pointer handle) {
        super(parent, handle);
        this.mxNDArrayEx = new NDArrayEx(this);
    }

    /**
     * Constructs an {@link NDArray} from a native handle and metadata (internal).
     *
     * @param parent the parent {@link MxResource} to manage the life circle of the {@link NDArray}
     * @param handle the pointer to the native NDArray memory
     * @param fmt the sparse format
     */
    NDArray(MxResource parent, Pointer handle, SparseFormat fmt) {
        this(parent, handle);
        this.sparseFormat = fmt;
    }

    /**
     * Creates an NDArray with the given Native Memory Pointer and parent MxResource.
     *
     * @param parent the parent {@link MxResource} instance
     * @param handle the array's native memory pointer
     * @return the created array
     */
    public static NDArray create(MxResource parent, Pointer handle) {
        return new NDArray(parent, handle);
    }

    /**
     * Creates an NDArray with the given Native Memory Pointer and parent MxResource.
     *
     * @param parent the parent {@link MxResource} instance
     * @param handle the array's native memory pointer
     * @param fmt the sparse format
     * @return the created array
     */
    public static NDArray create(MxResource parent, Pointer handle, SparseFormat fmt) {
        return new NDArray(parent, handle, fmt);
    }

    /**
     * Creates an uninitialized instance of {@link DataType#FLOAT32} {@link NDArray} with specified
     * parent {@link MxResource}, {@link Shape}, {@link Device} and {@code hasGradient}.
     *
     * @param parent the parent {@link MxResource}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, Shape shape, Device device) {
        return create(parent, shape, DataType.FLOAT32, device);
    }

    /**
     * Creates an uninitialized instance of {@link DataType#FLOAT32} {@link NDArray} with specified
     * parent {@link MxResource} and {@link Shape}.
     *
     * @param parent the parent {@link MxResource}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, Shape shape) {
        return create(parent, shape, DataType.FLOAT32, Device.defaultIfNull());
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified parent {@link
     * MxResource}, {@link Shape}, {@link DataType}, {@link Device} and {@code hasGradient}.
     *
     * @param parent the parent {@link MxResource}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @param hasGradient true if the gradient calculation is required for this {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(
            MxResource parent, Shape shape, DataType dataType, Device device, boolean hasGradient) {
        Pointer handle =
                JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), hasGradient);
        return new NDArray(parent, handle, device, shape, dataType, hasGradient);
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified parent {@link
     * MxResource}, {@link Shape}, {@link DataType}, {@link Device}.
     *
     * @param parent the parent {@link MxResource}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, Shape shape, DataType dataType, Device device) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        return new NDArray(parent, handle, Device.defaultIfNull(device), shape, dataType, false);
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} instance
     * @param data the {@link Number} that needs to be set
     * @return a new instance of {@link NDArray}
     * @throws IllegalArgumentException when the Type of data is not expected
     */
    public static NDArray create(MxResource parent, Number data) {
        if (data instanceof Integer) {
            return create(parent, data.intValue());
        } else if (data instanceof Float) {
            return create(parent, data.floatValue());
        } else if (data instanceof Double) {
            return create(parent, data.doubleValue());
        } else if (data instanceof Long) {
            return create(parent, data.longValue());
        } else if (data instanceof Byte) {
            return create(parent, data.byteValue());
        } else {
            throw new IllegalArgumentException("Short conversion not supported!");
        }
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and float
     * array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, float[] data, Shape shape) {
        return create(parent, FloatBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and int
     * array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, int[] data, Shape shape) {
        return create(parent, IntBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and
     * double array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, double[] data, Shape shape) {
        return create(parent, DoubleBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and long
     * array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, long[] data, Shape shape) {
        return create(parent, LongBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and byte
     * array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, byte[] data, Shape shape) {
        return create(parent, ByteBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and
     * boolean array.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the boolean array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, boolean[] data, Shape shape) {
        byte[] byteData = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            byteData[i] = (byte) (data[i] ? 1 : 0);
        }
        return create(parent, ByteBuffer.wrap(byteData), shape, DataType.BOOLEAN);
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, float data) {
        return create(parent, new float[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, int data) {
        return create(parent, new int[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the double data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, double data) {
        return create(parent, new double[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the long data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, long data) {
        return create(parent, new long[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the byte data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, byte data) {
        return create(parent, new byte[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the boolean data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, boolean data) {

        return create(parent, new boolean[] {data}, new Shape());
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, float[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, int[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, double[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, long[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, byte[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the bool array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, boolean[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, float[][] data) {
        FloatBuffer buffer = FloatBuffer.allocate(data.length * data[0].length);
        for (float[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, int[][] data) {
        IntBuffer buffer = IntBuffer.allocate(data.length * data[0].length);
        for (int[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, double[][] data) {
        DoubleBuffer buffer = DoubleBuffer.allocate(data.length * data[0].length);
        for (double[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, long[][] data) {
        LongBuffer buffer = LongBuffer.allocate(data.length * data[0].length);
        for (long[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} instance
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, byte[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * data[0].length);
        for (byte[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param parent the parent {@link MxResource} instance
     * @param data the boolean array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    public static NDArray create(MxResource parent, boolean[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * data[0].length);
        for (boolean[] d : data) {
            for (boolean b : d) {
                buffer.put((byte) (b ? 1 : 0));
            }
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length), DataType.BOOLEAN);
    }

    /**
     * Creates and initializes a {@link NDArray} with specified {@link Shape}.
     *
     * <p>{@link DataType} of the MxNDArray will determined by type of Buffer.
     *
     * @param parent the parent {@link MxResource} instance
     * @param data the data to initialize the {@code MxNDArray}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    static NDArray create(MxResource parent, Buffer data, Shape shape) {
        DataType dataType = DataType.fromBuffer(data);
        return create(parent, data, shape, dataType);
    }

    static NDArray create(MxResource parent, Buffer data, Shape shape, DataType dataType) {
        NDArray array = create(parent, shape, dataType, Device.defaultIfNull());
        array.set(data);
        return array;
    }

    /**
     * Returns the name of this {@code NDArray}.
     *
     * @return the name of this {@code NDArray}
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name of this {@code NDArray}.
     *
     * @param name of the {@code NDArray}
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the {@link DataType} of this {@code NDArray}.
     *
     * @return the {@link DataType} of this {@code NDArray}
     */
    public DataType getDataType() {
        if (this.dataType == null) {
            this.dataType = JnaUtils.getDataTypeOfNdArray(getHandle());
        }
        return this.dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (this.device == null) {
            this.device = JnaUtils.getDeviceOfNdArray(getHandle());
        }
        return this.device;
    }

    /**
     * Returns the {@link Shape} of this {@code NDArray}.
     *
     * @return the {@link Shape} of this {@code NDArray}
     */
    public Shape getShape() {
        if (this.shape == null) {
            this.shape = JnaUtils.getShapeOfNdArray(getHandle());
        }
        return this.shape;
    }

    /**
     * Returns the {@link SparseFormat} of this {@code NDArray}.
     *
     * @return the {@link SparseFormat} of this {@code NDArray}
     */
    public SparseFormat getSparseFormat() {
        if (this.sparseFormat == null) {
            this.sparseFormat = JnaUtils.getStorageType(getHandle());
        }
        return this.sparseFormat;
    }

    /**
     * Returns the version of this {@code NDArray}.
     *
     * @return the version of this {@code NDArray}
     */
    public Integer getVersion() {
        if (this.version == null) {
            this.version = JnaUtils.getVersion();
        }
        return this.version;
    }

    private NDArray duplicate(Shape shape, DataType dataType, Device device, String name) {
        // TODO get copy parameter
        NDArray array = create(getParent(), shape, dataType, device);
        array.setName(name);
        copyTo(array);
        return array;
    }

    /**
     * Returns a copy of this {@code NDArray}.
     *
     * @return a copy of this {@code NDArray}
     */
    NDArray duplicate() {
        NDArray array = create(getParent(), getShape(), getDataType(), getDevice());
        array.setName(getName());
        copyTo(array);
        return array;
    }

    /**
     * Moves this {@code NDArray} to a different {@link Device}.
     *
     * @param device the {@link Device} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link Device}
     */
    public NDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice()) && !copy) {
            return this;
        }
        return duplicate(getShape(), getDataType(), device, getName());
    }

    /**
     * Converts this {@code NDArray} to a different {@link DataType}.
     *
     * @param dataType the {@link DataType} to be set
     * @param copy set {@code true} if you want to return a copy of the Existing {@code NDArray}
     * @return the result {@code NDArray} with the new {@link DataType}
     */
    public NDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        return duplicate(getShape(), dataType, getDevice(), getName());
    }

    /**
     * Attaches a gradient {@code NDArray} to this {@code NDArray} and marks it. It is related to
     * training so will not be used here.
     *
     * @param requiresGrad if {@code NDArray} requires gradient or not
     */
    public void setRequiresGradient(boolean requiresGrad) {
        if ((requiresGrad && hasGradient()) || (!requiresGrad && !hasGradient())) {
            return;
        }
        NDArray grad = hasGradient() ? getGradient() : createGradient(getSparseFormat());
        // DJL go with write as only MXNet support GradReq
        int gradReqValue = requiresGrad ? GradReq.WRITE.getValue() : GradReq.NULL.getValue();
        IntBuffer gradReqBuffer = IntBuffer.allocate(1);
        gradReqBuffer.put(0, gradReqValue);
        JnaUtils.autogradMarkVariables(1, getHandle(), gradReqBuffer, grad.getHandle());
        hasGradient = requiresGrad;
        grad.close();
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     * @see #zeros(Shape, DataType, Device)
     */
    public NDArray zeros(Shape shape, DataType dataType) {
        return fill("_npi_zeros", shape, dataType);
    }

    /**
     * Creates an instance of {@link NDArray} with the same {@link Shape} and {@link DataType}
     * filled with zeros.
     *
     * @return a new instance of {@link NDArray}
     * @see #zeros(Shape, DataType, Device)
     */
    public NDArray zeros() {
        return zeros(getShape(), getDataType());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray zeros(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return zeros(shape, dataType);
        }
        return zeros(shape, dataType);
    }

    private NDArray createGradient(SparseFormat format) {
        try (NDArray zeros = this.zeros(getShape(), getDataType(), getDevice())) {
            return zeros.toSparse(format);
        }
    }

    private NDArray fill(String opName, Shape shape, DataType dataType) {
        OpParams params = new OpParams();
        if (shape == null) {
            throw new IllegalArgumentException("Shape is required for " + opName.substring(1));
        }
        params.addParam("shape", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke(getParent(), opName, params);
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param parent the parent {@link MxResource}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    public static NDArray ones(MxResource parent, Shape shape, DataType dataType, Device device) {
        return create(parent, shape, dataType, device).ones();
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray ones(Shape shape, DataType dataType) {
        return fill("_npi_ones", shape, dataType);
    }

    /**
     * Creates an instance of {@link NDArray} with same {@link Shape} and {@link DataType} filled
     * with ones.
     *
     * @return a new instance of {@link NDArray}
     */
    public NDArray ones() {
        return ones(getShape(), getDataType());
    }
    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray ones(Shape shape) {
        return ones(shape, DataType.FLOAT32);
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray ones(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return ones(shape, dataType);
        }
        return create(getParent(), shape, dataType, device).ones();
    }

    /**
     * Returns the gradient {@code NDArray} attached to this {@code NDArray}.
     *
     * @return the gradient {@code NDArray}
     * @throws IllegalStateException when hasGradient is false
     */
    public NDArray getGradient() {
        if (!hasGradient()) {
            throw new IllegalStateException(
                    "No gradient attached to this MxNDArray, please call array.requiredGradient()"
                            + "on your MxNDArray or block.setInitializer() on your Block");
        }
        Pointer pointer = JnaUtils.getGradient(getHandle());
        return create(getParent(), pointer);
    }

    /**
     * Returns true if the gradient calculation is required for this {@code NDArray}.
     *
     * @return true if the gradient calculation is required for this {@code NDArray} else false
     */
    public boolean hasGradient() {
        if (hasGradient == null) {
            Pointer pointer = JnaUtils.getGradient(getHandle());
            hasGradient = pointer != null;
        }
        return hasGradient;
    }

    /**
     * Returns an NDArray equal to this that stop gradient propagation through it.
     *
     * @return an NDArray equal to this that stops gradient propagation through it
     */
    public NDArray stopGradient() {
        Pointer pointer = JnaUtils.detachGradient(getHandle());
        return create(getParent(), pointer);
    }

    /**
     * Converts this {@code NDArray} to a String array.
     *
     * <p>This method is only applicable to the String typed NDArray and not for printing purpose
     *
     * @return Array of Strings
     */
    public String[] toStringArray() {
        throw new UnsupportedOperationException("String MxNDArray is not supported!");
    }

    /**
     * Converts this {@code NDArray} to a ByteBuffer.
     *
     * @return a ByteBuffer
     */
    public ByteBuffer toByteBuffer() {
        if (getSparseFormat() != SparseFormat.DENSE) {
            throw new IllegalStateException("Require Dense MxNDArray, actual " + getSparseFormat());
        }
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        ByteBuffer bb = NDSerializer.allocateDirect(Math.toIntExact(len));
        Pointer pointer = Native.getDirectBufferPointer(bb);
        JnaUtils.syncCopyToCPU(getHandle(), pointer, Math.toIntExact(product));
        return bb;
    }

    /**
     * Returns the total number of elements in this {@code MxNDArray}.
     *
     * @return the number of elements in this {@code MxNDArray}
     */
    long size() {
        return getShape().size();
    }

    long size(int axis) {
        return getShape().size(axis);
    }

    /**
     * Sets this {@code NDArray} value from {@link Buffer}.
     *
     * @param data the input buffered data
     */
    public void set(Buffer data) {
        int size = Math.toIntExact(size());
        if (data.remaining() < size) {
            throw new IllegalArgumentException(
                    "The MxNDArray size is: " + size + ", but buffer size is: " + data.remaining());
        }
        if (data.isDirect()) {
            JnaUtils.syncCopyFromCPU(getHandle(), data, size);
            return;
        }

        data.limit(size);
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        DataType inputType = DataType.fromBuffer(data);
        validate(inputType);

        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = NDSerializer.allocateDirect(size * numOfBytes);

        switch (inputType) {
            case FLOAT32:
                buf.asFloatBuffer().put((FloatBuffer) data);
                break;
            case FLOAT64:
                buf.asDoubleBuffer().put((DoubleBuffer) data);
                break;
            case UINT8:
            case INT8:
            case BOOLEAN:
                buf.put((ByteBuffer) data);
                break;
            case INT32:
                buf.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                buf.asLongBuffer().put((LongBuffer) data);
                break;
            case FLOAT16:
            default:
                throw new UnsupportedOperationException("data type is not supported!");
        }
        buf.rewind();
        JnaUtils.syncCopyFromCPU(getHandle(), buf, size);
    }

    private void validate(DataType inputType) {
        if (getDataType() != inputType
                && ((dataType != DataType.UINT8 && dataType != DataType.BOOLEAN)
                        || inputType != DataType.INT8)) {
            // Infer DataType from Buffer always return INT8, make this two special case that
            // allows set UINT8 and BOOL array with regular ByteBuffer.
            throw new IllegalStateException(
                    "DataType mismatch, required: " + dataType + ", actual: " + inputType);
        }
    }

    /**
     * Returns {@code true} if this {@code MxNDArray} is a scalar {@code MxNDArray} with empty
     * {@link Shape}.
     *
     * @return {@code true} if this {@code MxNDArray} is a scalar {@code MxNDArray} with empty
     *     {@link Shape}
     */
    boolean isScalar() {
        return getShape().isScalar();
    }

    /**
     * Returns {@code true} if all elements within this {@code NDArray} are non-zero or {@code
     * true}.
     *
     * @return {@code true} if all elements within this {@code NDArray} are non-zero or {@code true}
     */
    NDArray all() {
        // result of sum operation is int64 now
        return toType(DataType.BOOLEAN, false).sum().eq(size());
    }

    /**
     * Deep-copies the current {@code NDArray} to the one passed in.
     *
     * @param ndArray this {@code NDArray} prepared to be copied to
     */
    public void copyTo(NDArray ndArray) {

        Shape inShape = getShape();
        Shape destShape = ndArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        JnaUtils.op("_npi_copyto").invoke(new NDArray[] {this}, new NDArray[] {ndArray}, null);
    }

    NDArray booleanMask(NDArray index) {
        return booleanMask(index, 0);
    }

    /**
     * Returns portion of this {@code NDArray} given the index boolean {@code NDArray} along given
     * axis.
     *
     * @param index boolean {@code NDArray} mask
     * @param axis an integer that represents the axis of {@code NDArray} to mask from
     * @return the result {@code NDArray}
     */
    public NDArray booleanMask(NDArray index, int axis) {
        if (isScalar() || index.isScalar()) {
            throw new IllegalArgumentException("booleanMask didn't support scalar!");
        }
        // TODO remove reshape when MXNet numpy support multi-dim index
        // and boolean MxNDArray reshape
        Shape remainingDims = getShape().slice(index.getShape().dimension());
        // create a reshape array {-1, remainingDims}
        long[] reshape = new long[remainingDims.dimension() + 1];
        reshape[0] = -1;
        System.arraycopy(remainingDims.getShape(), 0, reshape, 1, remainingDims.dimension());
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        try (NDArray reshaped = this.reshape(new Shape(reshape));
                NDArray reshapedIndex = index.toType(DataType.INT32, false).reshape(-1);
                NDArray result =
                        invoke(
                                getParent(),
                                "_npi_boolean_mask",
                                new NDArray[] {reshaped, reshapedIndex},
                                params)) {
            return result.reshape(reshape);
        }
    }

    /**
     * Sets all elements outside the sequence to a constant value.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @param value the constant value to be set
     * @return the result {@code NDArray}
     */
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        if (getShape().dimension() < 2 || getShape().isScalar() || getShape().hasZeroDimension()) {
            throw new IllegalArgumentException(
                    "sequenceMask is not supported for MxNDArray with less than 2 dimensions");
        }
        Shape expectedSequenceLengthShape = new Shape(getShape().get(0));
        if (!sequenceLength.getShape().equals(expectedSequenceLengthShape)) {
            throw new IllegalArgumentException("SequenceLength must be of shape [batchSize]");
        }
        OpParams params = new OpParams();
        params.add("value", value);
        params.add("use_sequence_length", true);
        params.add("axis", 1);
        return invoke(getParent(), "_npx_sequence_mask", new NDList(this, sequenceLength), params)
                .head();
    }

    /**
     * Sets all elements outside the sequence to 0.
     *
     * <p>This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter {@code
     * sequenceLength} is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @return the result {@code NDArray}
     */
    public NDArray sequenceMask(NDArray sequenceLength) {
        return sequenceMask(sequenceLength, 0);
    }

    /**
     * Returns an {@code NDArray} of zeros with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * @return a {@code NDArray} filled with zeros
     */
    public NDArray zerosLike() {
        OpParams params = new OpParams();
        params.addParam("fill_value", 0);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    /**
     * Returns an {@code NDArray} of ones with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input {@code NDArray}.
     *
     * @return a {@code NDArray} filled with ones
     */
    public NDArray onesLike() {
        OpParams params = new OpParams();
        params.addParam("fill_value", 1);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    NDArray get(NDIndex index) {
        return getNDArrayInternal().getIndexer().get(this, index);
    }

    NDArray get(long... indices) {
        return get(new NDIndex(indices));
    }

    NDArray getScalar(long... indices) {
        NDArray value = get(new NDIndex(indices));
        if (value.size() != 1) {
            throw new IllegalArgumentException("The supplied Index does not produce a scalar");
        }
        return value;
    }

    boolean getBoolean(long... indices) {
        return getScalar(indices).toBooleanArray()[0];
    }

    /**
     * Returns {@code true} if all elements in this {@code NDArray} are equal to the {@link Number}.
     *
     * @param number the number to compare
     * @return the boolean result
     */
    public boolean contentEquals(Number number) {
        if (number == null) {
            return false;
        }
        try (NDArray result = eq(number)) {
            return result.all().getBoolean();
        }
    }

    /**
     * Returns {@code true} if all elements in this {@code NDArray} are equal to the other {@link
     * NDArray}.
     *
     * @param other the other {@code NDArray} to compare
     * @return the boolean result
     */
    public boolean contentEquals(NDArray other) {
        if (other == null || (!shapeEquals(other))) {
            return false;
        }
        if (getDataType() != other.getDataType()) {
            return false;
        }
        try (NDArray result = eq(other).toType(DataType.INT32, false)) {
            return result.all().getBoolean();
        }
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Equals" comparison.
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    public NDArray eq(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_equal_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Equals" comparison
     */
    public NDArray eq(NDArray other) {
        return invoke(getParent(), "_npi_equal", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * @param n the number to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    public NDArray neq(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_not_equal_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Not equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Not equals" comparison
     */
    public NDArray neq(NDArray other) {
        return invoke(getParent(), "_npi_not_equal", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Greater" comparison
     */
    public NDArray gt(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater Than" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wis "Greater Than" comparison
     */
    public NDArray gt(NDArray other) {
        return invoke(getParent(), "_npi_greater", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Greater or equals" comparison
     */
    public NDArray gte(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_equal_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Greater or equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for "Greater or equals" comparison
     */
    public NDArray gte(NDArray other) {
        return invoke(getParent(), "_npi_greater_equal", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    public NDArray lt(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less" comparison
     */
    public NDArray lt(NDArray other) {
        return invoke(getParent(), "_npi_less", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * @param other the number to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    public NDArray lte(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_equal_scalar", this, params);
    }

    /**
     * Returns the boolean {@code NDArray} for element-wise "Less or equals" comparison.
     *
     * @param other the {@code NDArray} to compare
     * @return the boolean {@code NDArray} for element-wise "Less or equals" comparison
     */
    public NDArray lte(NDArray other) {
        return invoke(getParent(), "_npi_less_equal", new NDArray[] {this, other}, null);
    }

    /**
     * Adds a number to this {@code NDArray} element-wise.
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    public NDArray add(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_add_scalar", this, params);
    }

    /**
     * Adds other {@code NDArray}s to this {@code NDArray} element-wise.
     *
     * @param other the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    public NDArray add(NDArray other) {
        return invoke(getParent(), "_npi_add", new NDArray[] {this, other}, null);
    }

    /**
     * Subtracts a number from this {@code NDArray} element-wise.
     *
     * @param n the number to subtract from
     * @return the result {@code NDArray}
     */
    public NDArray sub(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_subtract_scalar", this, params);
    }

    /**
     * Subtracts the other {@code NDArray} from this {@code NDArray} element-wise.
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    public NDArray sub(NDArray other) {
        return invoke(getParent(), "_npi_subtract", new NDArray[] {this, other}, null);
    }

    /**
     * Multiplies this {@code NDArray} by a number element-wise.
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    public NDArray mul(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_multiply_scalar", this, params);
    }

    /**
     * Multiplies this {@code NDArray} by other {@code NDArray}s element-wise.
     *
     * @param other the other {@code NDArray}s to multiply by
     * @return the result {@code NDArray}
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    public NDArray mul(NDArray other) {
        return invoke(getParent(), "_npi_multiply", new NDArray[] {this, other}, null);
    }

    /**
     * Divides this {@code NDArray} by a number element-wise.
     *
     * @param n the number to divide by
     * @return the result {@code NDArray}
     */
    public NDArray div(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_true_divide_scalar", this, params);
    }

    /**
     * Divides this {@code NDArray} by the other {@code NDArray} element-wise.
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result {@code NDArray}
     */
    public NDArray div(NDArray other) {
        return invoke(getParent(), "_npi_true_divide", new NDArray[] {this, other}, null);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    public NDArray mod(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_mod_scalar", this, params);
    }

    /**
     * Returns element-wise remainder of division.
     *
     * @param other the divisor {@code NDArray}
     * @return the result {@code NDArray}
     */
    public NDArray mod(NDArray other) {
        return invoke(getParent(), "_npi_mod", new NDArray[] {this, other}, null);
    }

    /**
     * Takes the power of this {@code NDArray} with a number element-wise.
     *
     * @param n the number to take the power with
     * @return the result {@code NDArray}
     */
    public NDArray pow(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_power_scalar", this, params);
    }

    /**
     * Takes the power of this {@code NDArray} with the other {@code NDArray} element-wise.
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    public NDArray pow(NDArray other) {
        return invoke(getParent(), "_npi_power", new NDArray[] {this, other}, null);
    }

    /**
     * Adds a number to this {@code NDArray} element-wise in place.
     *
     * @param n the number to add
     * @return the result {@code NDArray}
     */
    public NDArray addi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_add_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Adds other {@code NDArray}s to this {@code NDArray} element-wise in place.
     *
     * @param other the other {@code NDArray}s to add
     * @return the result {@code NDArray}
     */
    public NDArray addi(NDArray other) {
        invoke("_npi_add", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Subtracts a number from this {@code NDArray} element-wise in place.
     *
     * @param n the number to subtract
     * @return the result {@code NDArray}
     */
    public NDArray subi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_subtract_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Subtracts the other {@code NDArray} from this {@code NDArray} element-wise in place.
     *
     * @param other the other {@code NDArray} to subtract from
     * @return the result {@code NDArray}
     */
    public NDArray subi(NDArray other) {
        invoke("_npi_subtract", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Multiplies this {@code NDArray} by a number element-wise in place.
     *
     * @param n the number to multiply by
     * @return the result {@code NDArray}
     */
    public NDArray muli(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_multiply_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Multiplies this {@code NDArray} by other {@code NDArray} element-wise in place.
     *
     * @param other the other NDArrays to multiply with
     * @return the result {@code NDArray}
     */
    public NDArray muli(NDArray other) {
        invoke("_npi_multiply", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Divides this {@code NDArray} by a number element-wise in place.
     *
     * @param n the number to divide values by
     * @return the array after applying division operation
     */
    public NDArray divi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_true_divide_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Divides this {@code NDArray} by the other {@code NDArray} element-wise in place.
     *
     * @param other the other {@code NDArray} to divide by
     * @return the result of the divide
     */
    public NDArray divi(NDArray other) {
        invoke("_npi_true_divide", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Returns element-wise remainder of division in place.
     *
     * @param n the divisor number
     * @return the result {@code NDArray}
     */
    public NDArray modi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_mod_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Returns in place element-wise remainder of division in place.
     *
     * @param other the divisor {@code NDArray}
     * @return the result of the divide
     */
    public NDArray modi(NDArray other) {
        invoke("_npi_mod", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Takes the power of this {@code NDArray} with a number element-wise in place.
     *
     * @param n the number to raise the power to
     * @return the result {@code NDArray}
     */
    public NDArray powi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_power_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /**
     * Takes the power of this {@code NDArray} with the other {@code NDArray} element-wise in place.
     *
     * @param other the other {@code NDArray} to take the power with
     * @return the result {@code NDArray}
     */
    public NDArray powi(NDArray other) {
        invoke("_npi_power", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Returns the element-wise sign.
     *
     * @return the result {@code NDArray}
     */
    public NDArray sign() {
        return invoke(getParent(), "_npi_sign", this, null);
    }

    /**
     * Returns the element-wise sign in-place.
     *
     * @return the result {@code NDArray}
     */
    public NDArray signi() {
        invoke("_npi_sign", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Returns the maximum of this {@code NDArray} and a number element-wise.
     *
     * @param n the number to be compared
     * @return the maximum of this {@code NDArray} and a number element-wise
     */
    public NDArray maximum(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_maximum_scalar", this, params);
    }

    /**
     * Returns the maximum of this {@code NDArray} and the other {@code NDArray} element-wise.
     *
     * @param other the {@code NDArray} to be compared
     * @return the maximum of this {@code NDArray} and the other {@code NDArray} element-wise
     */
    public NDArray maximum(NDArray other) {
        return invoke(getParent(), "_npi_maximum", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the minimum of this {@code NDArray} and a number element-wise.
     *
     * @param n the number to be compared
     * @return the minimum of this {@code NDArray} and a number element-wise
     */
    public NDArray minimum(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_minimum_scalar", this, params);
    }

    /**
     * Returns the maximum of this {@code NDArray} and the other {@code NDArray} element-wise.
     *
     * @param other the {@code NDArray} to be compared
     * @return the maximum of this {@code NDArray} and the other {@code NDArray} element-wise
     */
    public NDArray minimum(NDArray other) {
        return invoke(getParent(), "_npi_minimum", new NDArray[] {this, other}, null);
    }

    /**
     * Returns the numerical negative {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray neg() {
        return invoke(getParent(), "_npi_negative", this, null);
    }

    /**
     * Returns the numerical negative {@code NDArray} element-wise in place.
     *
     * @return the result {@code NDArray}
     */
    public NDArray negi() {
        invoke("_npi_negative", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    /**
     * Returns the absolute value of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray abs() {
        return invoke(getParent(), "_npi_absolute", this, null);
    }

    /**
     * Returns the square of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray square() {
        return invoke(getParent(), "_npi_square", this, null);
    }

    /**
     * Returns the square root of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray sqrt() {
        return invoke(getParent(), "_npi_sqrt", this, null);
    }

    /**
     * Returns the cube-root of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray cbrt() {
        return invoke(getParent(), "_npi_cbrt", this, null);
    }

    /**
     * Returns the floor of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray floor() {
        return invoke(getParent(), "_npi_floor", this, null);
    }

    /**
     * Returns the ceiling of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray ceil() {
        return invoke(getParent(), "_npi_ceil", this, null);
    }

    /**
     * Returns the round of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray round() {
        return invoke(getParent(), "round", this, null);
    }

    /**
     * Returns the truncated value of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray trunc() {
        return invoke(getParent(), "_npi_trunc", this, null);
    }

    /**
     * Returns the exponential value of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray exp() {
        return invoke(getParent(), "_npi_exp", this, null);
    }

    /**
     * Returns the natural logarithmic value of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray log() {
        return invoke(getParent(), "_npi_log", this, null);
    }

    /**
     * Returns the base 10 logarithm of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray log10() {
        return invoke(getParent(), "_npi_log10", this, null);
    }

    /**
     * Returns the base 2 logarithm of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray log2() {
        return invoke(getParent(), "_npi_log2", this, null);
    }

    /**
     * Returns the trigonometric sine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray sin() {
        return invoke(getParent(), "_npi_sin", this, null);
    }

    /**
     * Returns the trigonometric cosine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray cos() {
        return invoke(getParent(), "_npi_cos", this, null);
    }

    /**
     * Returns the trigonometric tangent of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray tan() {
        return invoke(getParent(), "_npi_tan", this, null);
    }

    /**
     * Returns the inverse trigonometric sine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray asin() {
        return invoke(getParent(), "_npi_arcsin", this, null);
    }

    /**
     * Returns the inverse trigonometric cosine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray acos() {
        return invoke(getParent(), "_npi_arccos", this, null);
    }

    /**
     * Returns the inverse trigonometric tangent of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray atan() {
        return invoke(getParent(), "_npi_arctan", this, null);
    }

    /**
     * Returns the hyperbolic sine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray sinh() {
        return invoke(getParent(), "_npi_sinh", this, null);
    }

    /**
     * Returns the hyperbolic cosine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray cosh() {
        return invoke(getParent(), "_npi_cosh", this, null);
    }

    /**
     * Returns the hyperbolic tangent of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray tanh() {
        return invoke(getParent(), "_npi_tanh", this, null);
    }

    /**
     * Returns the inverse hyperbolic sine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray asinh() {
        return invoke(getParent(), "_npi_arcsinh", this, null);
    }

    /**
     * Returns the inverse hyperbolic cosine of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray acosh() {
        return invoke(getParent(), "_npi_arccosh", this, null);
    }

    /**
     * Returns the inverse hyperbolic tangent of this {@code NDArray} element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray atanh() {
        return invoke(getParent(), "_npi_arctanh", this, null);
    }

    /**
     * Converts this {@code NDArray} from radians to degrees element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray toDegrees() {
        return invoke(getParent(), "_npi_degrees", this, null);
    }

    /**
     * Converts this {@code NDArray} from degrees to radians element-wise.
     *
     * @return the result {@code NDArray}
     */
    public NDArray toRadians() {
        return invoke(getParent(), "_npi_radians", this, null);
    }

    /**
     * Returns the maximum of this {@code NDArray}.
     *
     * @return the maximum of this {@code NDArray}
     */
    public NDArray max() {
        return invoke(getParent(), "_np_max", this, null);
    }

    /**
     * Returns the maximum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the maximum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the max
     * @see NDArray#max(int[], boolean)
     */
    public NDArray max(int[] axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_max", this, params);
    }

    /**
     * Returns the maximum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array.
     * @return the maximum of this {@code NDArray}
     */
    public NDArray max(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_max", this, params);
    }

    /**
     * Returns the minimum of this {@code NDArray}.
     *
     * @return the minimum of this {@code NDArray}
     */
    public NDArray min() {
        return invoke(getParent(), "_np_min", this, null);
    }

    /**
     * Returns the minimum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the minimum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the min
     * @see NDArray#min(int[], boolean)
     */
    public NDArray min(int[] axes) {
        return min(axes, false);
    }

    /**
     * Returns the minimum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the minimum of this {@code NDArray}
     */
    public NDArray min(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_min", this, params);
    }

    /**
     * Returns the sum of this {@code NDArray}.
     *
     * @return the sum of this {@code NDArray}
     */
    public NDArray sum() {
        // TODO current windows doesn't support boolean MxNDArray
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            DataType target = getDataType();
            if (!target.isFloating()) {
                try (NDArray thisArr = toType(DataType.FLOAT32, false)) {
                    if (target == DataType.BOOLEAN) {
                        target = DataType.INT64;
                    }
                    try (NDArray array = invoke(getParent(), "_np_sum", thisArr, null)) {
                        return array.toType(target, false);
                    }
                }
            }
        }
        return invoke(getParent(), "_np_sum", this, null);
    }

    /**
     * Returns the sum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the sum of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the sum
     * @see NDArray#sum(int[], boolean)
     */
    public NDArray sum(int[] axes) {
        return sum(axes, false);
    }

    /**
     * Returns the sum of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the sum of this {@code NDArray}
     */
    public NDArray sum(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_sum", this, params);
    }

    /**
     * Returns the product of this {@code NDArray}.
     *
     * @return the product of this {@code NDArray}
     */
    public NDArray prod() {
        return invoke(getParent(), "_np_prod", this, null);
    }

    /**
     * Returns the product of this {@code NDArray} elements over the given axes.
     *
     * @param axes the axes along which to operate
     * @return the product of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the prod
     * @see NDArray#prod(int[], boolean)
     */
    NDArray prod(int[] axes) {
        return prod(axes, false);
    }

    /**
     * Returns the product of this {@code NDArray} elements over the given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the product of this {@code NDArray}
     */
    public NDArray prod(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_prod", this, params);
    }

    /**
     * Returns the average of this {@code NDArray}.
     *
     * @return the average of this {@code NDArray}
     */
    public NDArray mean() {
        return invoke(getParent(), "_npi_mean", this, null);
    }

    /**
     * Returns the average of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @return the average of this {@code NDArray} with the specified axes removed from the Shape
     *     containing the mean
     * @see NDArray#mean(int[], boolean)
     */
    public NDArray mean(int[] axes) {
        return mean(axes, false);
    }

    /**
     * Returns the average of this {@code NDArray} along given axes.
     *
     * @param axes the axes along which to operate
     * @param keepDims {@code true} to keep the specified axes as size 1 in the output array, {@code
     *     false} to squeeze the values out of the output array
     * @return the average of this {@code NDArray}
     */
    public NDArray mean(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_mean", this, params);
    }

    /**
     * Rotates an array by 90 degrees in the plane specified by axes.
     *
     * @param times Number of times the array is rotated by 90 degrees.
     * @param axes The array is rotated in the plane defined by the axes. Axes must be different.
     * @return the rotated NDArray
     */
    public NDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        OpParams params = new OpParams();
        params.addTupleParam("axes", axes);
        params.addParam("k", times);
        return invoke(getParent(), "_npi_rot90", this, params);
    }

    /**
     * Returns the sum along diagonals of this {@code NDArray}.
     *
     * <p>If this {@code NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this {@code NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The {@link Shape} of the resulting array is the same as
     * this {@code NDArray} with axis1 and axis2 removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @param axis1 axes to be used as the first axis of the 2-D sub-arrays from which the diagonals
     *     should be taken
     * @param axis2 axes to be used as the second axis of the 2-D sub-arrays from which the
     *     diagonals should be taken
     * @return the sum along diagonals of this {@code NDArray}
     */
    public NDArray trace(int offset, int axis1, int axis2) {
        OpParams params = new OpParams();
        params.addParam("offset", offset);
        params.addParam("axis1", axis1);
        params.addParam("axis2", axis2);
        return invoke(getParent(), "_np_trace", this, params);
    }

    /**
     * Returns the sum along diagonals of this {@code NDArray}.
     *
     * <p>If this {@code NDArray} is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this {@code NDArray} has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The {@link Shape} of the resulting array is the same as
     * this {@code NDArray} with axis1 and axis2 removed.
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     *     negative.
     * @return the sum along diagonals of this {@code NDArray}
     */
    public NDArray trace(int offset) {
        return trace(offset, 0, 1);
    }

    /**
     * Splits this {@code NDArray} into multiple sub{@code NDArray}s given sections along the given
     * axis.
     *
     * @param indices this {@code NDArray} will be divided into N (sections) equal arrays along axis
     * @param axis the axis to split along
     * @return an {@link NDList} with numOutputs {@code NDArray}s with {@link Shape} {@code
     *     (this.shape.axis /= axis) }
     * @throws IllegalArgumentException thrown if the numOutputs does not equally divide the given
     *     axis
     */
    public NDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(this);
        }
        OpParams params = new OpParams();
        // follow the numpy behavior
        if (indices[0] != 0) {
            long[] tempIndices = new long[indices.length + 1];
            tempIndices[0] = 0;
            System.arraycopy(indices, 0, tempIndices, 1, indices.length);
            indices = tempIndices;
        }
        params.addTupleParam("indices", indices);
        params.addParam("axis", axis);
        params.addParam("squeeze_axis", false);
        return invoke(getParent(), "_npi_split", new NDList(this), params);
    }

    /**
     * Flattens this {@code NDArray} into a 1-D {@code NDArray} in row-major order.
     *
     * <p>To flatten in column-major order, first transpose this {@code NDArray}
     *
     * @return a 1-D {@code NDArray} of equal size
     */
    public NDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    /**
     * Reshapes this {@code NDArray} to the given {@link Shape}.
     *
     * <p>You can reshape it to match another NDArray by calling {@code a.reshape(b.getShape()) }
     *
     * @param shape the {@link Shape} to reshape into. Must have equal size to the current shape
     * @return a reshaped {@code NDArray}
     * @throws IllegalArgumentException thrown if the given {@link Shape} does not match the size of
     *     the current shape
     */
    public NDArray reshape(Shape shape) {
        OpParams params = new OpParams();
        params.addParam("newshape", shape);
        return invoke(getParent(), "_np_reshape", this, params);
    }

    /**
     * Reshapes this {@code NDArray} to the given {@link Shape}.
     *
     * @param newShape the long array to reshape into. Must have equal size to the current shape
     * @return a reshaped {@code NDArray}
     * @throws IllegalArgumentException thrown if the given {@link Shape} does not match the size of
     *     the current shape
     */
    public NDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    /**
     * Expands the {@link Shape} of a {@code NDArray}.
     *
     * <p>Inserts a new axis that will appear at the axis position in the expanded {@code NDArray}
     * shape.
     *
     * @param axis the position in the expanded axes where the new axis is placed
     * @return the result {@code NDArray}. The number of dimensions is one greater than that of the
     *     {@code NDArray}
     */
    public NDArray expandDims(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_expand_dims", this, params);
    }

    /**
     * Removes all singleton dimensions from this {@code NDArray} {@link Shape}.
     *
     * @return a result {@code NDArray} of same size and data without singleton dimensions
     */
    public NDArray squeeze() {
        return invoke(getParent(), "_np_squeeze", this, null);
    }

    /**
     * Removes singleton dimensions at the given axes.
     *
     * <p>Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     *   [1.],
     *   [2.],
     *  ],
     * ]
     * jshell&gt; array.squeeze(new int[] {0, 2});
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     * </pre>
     *
     * @param axes the axes at which to remove the singleton dimensions
     * @return a result {@code NDArray} of same size and data without the axes at part of the shape
     * @throws IllegalArgumentException thrown if any of the given axes are not a singleton
     *     dimension
     */
    public NDArray squeeze(int[] axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_squeeze", this, params);
    }

    /**
     * Returns the truth value of this {@code NDArray} AND the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical AND operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    public NDArray logicalAnd(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        NDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_and", new NDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    /**
     * Computes the truth value of this {@code NDArray} OR the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical OR operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    public NDArray logicalOr(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        NDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_or", new NDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    /**
     * Computes the truth value of this {@code NDArray} XOR the other {@code NDArray} element-wise.
     *
     * <p>The shapes of this {@code NDArray} and the other {@code NDArray} must be broadcastable.
     *
     * @param other the other {@code NDArray} to operate on
     * @return the boolean {@code NDArray} of the logical XOR operation applied to the elements of
     *     this {@code NDArray} and the other {@code NDArray}
     */
    public NDArray logicalXor(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        NDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_xor", new NDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    /**
     * Computes the truth value of NOT this {@code NDArray} element-wise.
     *
     * @return the boolean {@code NDArray}
     */
    public NDArray logicalNot() {
        return invoke(getParent(), "_npi_logical_not", this, null);
    }

    /**
     * Returns the indices that would sort this {@code NDArray} given the axis.
     *
     * <p>Perform an indirect sort along the given axis. It returns a {@code NDArray} of indices of
     * the same {@link Shape} as this {@code NDArray}.
     *
     * @param axis the axis to sort along
     * @param ascending whether to sort ascending
     * @return a {@code NDArray} of indices corresponding to elements in this {@code NDArray} on the
     *     axis, the output DataType is always {@link DataType#INT64}
     */
    public NDArray argSort(int axis, boolean ascending) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT64);
        return invoke(getParent(), "_npi_argsort", this, params);
    }

    /**
     * Sorts the flattened {@code NDArray}.
     *
     * @param axis the axis to sort along
     * @return the sorted {@code NDArray}
     */
    public NDArray sort(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_sort", this, params);
    }

    /**
     * Sorts the flattened {@code NDArray}.
     *
     * @return the sorted {@code NDArray}
     */
    public NDArray sort() {
        return invoke(getParent(), "_npi_sort", this, null);
    }

    /**
     * Applies the softmax function along the given axis.
     *
     * @param axis the axis along which to apply
     * @return the result {@code NDArray}
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a>
     * @see NDArray#softmax(int)
     */
    public NDArray softmax(int axis) {
        // MXNet softmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_softmax", this, params);
    }

    /**
     * Applies the softmax function followed by a logarithm.
     *
     * <p>Mathematically equivalent to calling softmax and then log. This single operator is faster
     * than calling two operators and numerically more stable when computing gradients.
     *
     * @param axis the axis along which to apply
     * @return the result {@code NDArray}
     */
    public NDArray logSoftmax(int axis) {
        // MXNet logsoftmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_log_softmax", this, params);
    }

    /**
     * Returns the cumulative sum of the elements in the flattened {@code NDArray}.
     *
     * @return the cumulative sum of the elements in the flattened {@code NDArray}
     */
    public NDArray cumSum() {
        return invoke(getParent(), "_np_cumsum", this, null);
    }

    /**
     * Return the cumulative sum of the elements along a given axis.
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified axis
     */
    public NDArray cumSum(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_np_cumsum", this, params);
    }

    /**
     * Replace the handle of the NDArray with the other. The NDArray used for replacement will be
     * killed.
     *
     * <p>Please use with caution, this method will make the input argument unusable.
     *
     * @param replaced the handle provider that will be killed
     */
    public void intern(NDArray replaced) {
        NDArray arr = replaced;
        Pointer oldHandle = handle.getAndSet(arr.handle.getAndSet(null));
        JnaUtils.waitToRead(oldHandle);
        JnaUtils.freeNdArray(oldHandle);
        // dereference old ndarray
        arr.close();
    }

    /**
     * Returns the boolean {@code NDArray} with value {@code true} where this {@code NDArray}'s
     * entries are infinite, or {@code false} where they are not infinite.
     *
     * @return the boolean {@code NDArray} with value {@code true} if this {@code NDArray}'s entries
     *     are infinite
     */
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /**
     * Returns the boolean {@code NDArray} with value {@code true} where this {@code NDArray}'s
     * entries are NaN, or {@code false} where they are not NaN.
     *
     * @return the boolean {@code NDArray} with value {@code true} if this {@code NDArray}'s {@link
     *     NDArray} are NaN
     */
    public NDArray isNaN() {
        return invoke(getParent(), "_npi_isnan", this, null);
    }

    /**
     * Returns a dense representation of the sparse {@code NDArray}.
     *
     * @return the result {@code NDArray}
     */
    public NDArray toDense() {
        if (!isSparse()) {
            return duplicate();
        }
        return castStorage(SparseFormat.DENSE);
    }

    /**
     * Returns a sparse representation of {@code NDArray}.
     *
     * @param fmt the {@link SparseFormat} of this {@code NDArray}
     * @return the result {@code NDArray}
     */
    public NDArray toSparse(SparseFormat fmt) {
        if (fmt != SparseFormat.DENSE
                && fmt != SparseFormat.CSR
                && fmt != SparseFormat.ROW_SPARSE) {
            throw new UnsupportedOperationException(fmt + " is not supported");
        }
        if (fmt == getSparseFormat()) {
            return duplicate();
        }
        return castStorage(fmt);
    }

    private NDArray castStorage(SparseFormat fmt) {
        OpParams params = new OpParams();
        params.setParam("stype", fmt.getType());
        return invoke(getParent(), "cast_storage", this, params);
    }

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given
     * repeats.
     *
     * @param repeats the number of times to repeat for each dimension
     * @return a NDArray that has been tiled
     */
    public NDArray tile(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given by
     * repeats.
     *
     * @param repeats the number of times to repeat along each axis
     * @return a {@code NDArray} that has been tiled
     */
    public NDArray tile(long[] repeats) {
        OpParams params = new OpParams();
        params.addTupleParam("reps", repeats);
        return invoke(getParent(), "_npi_tile", this, params);
    }

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times given by
     * repeats along given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return a {@code NDArray} that has been tiled
     * @throws IllegalArgumentException thrown for invalid axis
     */
    public NDArray tile(int axis, long repeats) {
        // scalar
        if (isScalar()) {
            throw new IllegalArgumentException("scalar didn't support specifying axis");
        }
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return tile(repeatsArray);
    }

    /**
     * Constructs a {@code NDArray} by repeating this {@code NDArray} the number of times to match
     * the desired shape.
     *
     * <p>If the desired {@link Shape}has fewer dimensions than this {@code NDArray}, it will tile
     * against the last axis.
     *
     * @param desiredShape the {@link Shape}that should be converted to
     * @return a {@code NDArray} that has been tiled
     */
    public NDArray tile(Shape desiredShape) {
        return tile(repeatsToMatchShape(desiredShape));
    }

    private int withAxis(int axis) {
        return Math.floorMod(axis, getShape().dimension());
    }

    private long[] repeatsToMatchShape(Shape desiredShape) {
        Shape curShape = getShape();
        int dimension = curShape.dimension();
        if (desiredShape.dimension() > dimension) {
            throw new IllegalArgumentException("The desired shape has too many dimensions");
        }
        if (desiredShape.dimension() < dimension) {
            int additionalDimensions = dimension - desiredShape.dimension();
            desiredShape = curShape.slice(0, additionalDimensions).addAll(desiredShape);
        }
        long[] repeats = new long[dimension];
        for (int i = 0; i < dimension; i++) {
            if (curShape.get(i) == 0 || desiredShape.get(i) % curShape.get(i) != 0) {
                throw new IllegalArgumentException(
                        "The desired shape is not a multiple of the original shape");
            }
            repeats[i] = Math.round(Math.ceil((double) desiredShape.get(i) / curShape.get(i)));
        }
        return repeats;
    }

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats.
     *
     * @param repeats the number of times to repeat for each axis
     * @return an {@code NDArray} that has been repeated
     */
    public NDArray repeat(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return repeat(repeatsArray);
    }

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats along given axis.
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return an {@code NDArray} that has been repeated
     * @throws IllegalArgumentException thrown for invalid axis
     */
    public NDArray repeat(int axis, long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return repeat(repeatsArray);
    }

    /**
     * Repeats element of this {@code NDArray} the number of times given repeats along each axis.
     *
     * @param repeats the number of times to repeat along each axis
     * @return a {@code NDArray} that has been repeated
     */
    public NDArray repeat(long[] repeats) {
        // TODO get rid of for loop once bug in MXNet np.repeat is fixed
        NDArray array = this;
        int baseAxis = getShape().dimension() - repeats.length;
        for (int i = 0; i < repeats.length; i++) {
            if (repeats[i] > 1) {
                NDArray previousArray = array;
                OpParams params = new OpParams();
                params.addParam("repeats", repeats[i]);
                params.addParam("axis", baseAxis + i);
                array = invoke(getParent(), "_np_repeat", array, params);
                if (previousArray != this) {
                    previousArray.close();
                }
            }
        }
        return array;
    }

    /**
     * Repeats element of this {@code NDArray} to match the desired shape.
     *
     * <p>If the desired {@link Shape} has fewer dimensions that the array, it will repeat against
     * the last axis.
     *
     * @param desiredShape the {@link Shape} that should be converted to
     * @return an {@code NDArray} that has been repeated
     */
    public NDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    /**
     * Dot product of this {@code NDArray} and the other {@code NDArray}.
     *
     * <ul>
     *   <li>If both this {@code NDArray} and the other {@code NDArray} are 1-D {@code NDArray}s, it
     *       is inner product of vectors (without complex conjugation).
     *   <li>If both this {@code NDArray} and the other {@code NDArray} are 2-D {@code NDArray}s, it
     *       is matrix multiplication.
     *   <li>If either this {@code NDArray} or the other {@code NDArray} is 0-D {@code NDArray}
     *       (scalar), it is equivalent to mul.
     *   <li>If this {@code NDArray} is N-D {@code NDArray} and the other {@code NDArray} is 1-D
     *       {@code NDArray}, it is a sum product over the last axis of those.
     *   <li>If this {@code NDArray} is N-D {@code NDArray} and the other {@code NDArray} is M-D
     *       {@code NDArray}(where M&gt;&#61;2), it is a sum product over the last axis of this
     *       {@code NDArray} and the second-to-last axis of the other {@code NDArray}
     * </ul>
     *
     * @param other the other {@code NDArray} to perform dot product with
     * @return the result {@code NDArray}
     */
    public NDArray dot(NDArray other) {
        return invoke(getParent(), "_np_dot", new NDArray[] {this, other}, null);
    }

    /**
     * Product matrix of this {@code NDArray} and the other {@code NDArray}.
     *
     * @param other the other {@code NDArray} to perform matrix product with
     * @return the result {@code NDArray}
     */
    public NDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        return invoke(getParent(), "_npi_matmul", new NDArray[] {this, other}, null);
    }

    /**
     * Clips (limit) the values in this {@code NDArray}.
     *
     * <p>Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values
     * larger than 1 become 1.
     *
     * @param min the minimum value
     * @param max the maximum value
     * @return an {@code NDArray} with the elements of this {@code NDArray}, but where values &lt;
     *     min are replaced with min, and those &gt; max with max
     */
    public NDArray clip(Number min, Number max) {
        OpParams params = new OpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return invoke(getParent(), "_npi_clip", this, params);
    }

    /**
     * Interchanges two axes of this {@code NDArray}.
     *
     * @param axis1 the first axis
     * @param axis2 the second axis
     * @return the swapped axes {@code NDArray}
     */
    public NDArray swapAxes(int axis1, int axis2) {
        OpParams params = new OpParams();
        params.addParam("dim1", axis1);
        params.addParam("dim2", axis2);
        return invoke(getParent(), "_npi_swapaxes", this, params);
    }

    /**
     * Returns the reverse order of elements in an array along the given axis.
     *
     * <p>The shape of the array is preserved, but the elements are reordered.
     *
     * @param axes the axes to flip on
     * @return the newly flipped array
     */
    public NDArray flip(int... axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_npi_flip", this, params);
    }

    /**
     * Returns this {@code NDArray} with axes transposed.
     *
     * @return the newly permuted array
     */
    public NDArray transpose() {
        return invoke(getParent(), "_np_transpose", this, null);
    }

    /**
     * Returns this {@code NDArray} with given axes transposed.
     *
     * @param dimensions the axes to swap to
     * @return the transposed {@code NDArray}
     * @throws IllegalArgumentException thrown when passing a axis that is greater than the actual
     *     number of dimensions
     */
    public NDArray transpose(int... dimensions) {
        if (Arrays.stream(dimensions).anyMatch(d -> d < 0)) {
            throw new UnsupportedOperationException(
                    "Passing -1 for broadcasting the dimension is not currently supported");
        }
        if (!Arrays.equals(
                Arrays.stream(dimensions).sorted().toArray(),
                IntStream.range(0, getShape().dimension()).toArray())) {
            throw new IllegalArgumentException(
                    "You must include each of the dimensions from 0 until "
                            + getShape().dimension());
        }
        OpParams params = new OpParams();
        params.addTupleParam("axes", dimensions);
        return invoke(getParent(), "_np_transpose", this, params);
    }

    /**
     * Broadcasts this {@code NDArray} to be the given shape.
     *
     * @param shape the new {@link Shape} of this {@code NDArray}
     * @return the broadcasted {@code NDArray}
     */
    public NDArray broadcast(Shape shape) {
        OpParams params = new OpParams();
        params.setShape(shape);
        return invoke(getParent(), "_npi_broadcast_to", this, params);
    }

    /**
     * Returns the indices of the maximum values into the flattened {@code NDArray}.
     *
     * @return a {@code NDArray} containing indices
     */
    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmax", this, null);
    }

    /**
     * Returns the indices of the maximum values along given axis.
     *
     * @param axis the axis along which to find maximum values
     * @return a {@code NDArray} containing indices
     */
    public NDArray argMax(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmax", this, params);
    }

    /**
     * Returns the indices of the minimum values into the flattened {@code NDArray}.
     *
     * @return a {@code NDArray} containing indices
     */
    public NDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmin", this, null);
    }

    /**
     * Returns the indices of the minimum values along given axis.
     *
     * @param axis the axis along which to find minimum values
     * @return a {@code NDArray} containing indices
     */
    public NDArray argMin(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmin", this, params);
    }

    /**
     * Returns percentile for this {@code NDArray}.
     *
     * @param percentile the target percentile in range of 0..100
     * @return the result {@code NDArray}
     */
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /**
     * Returns median along given dimension(s).
     *
     * @param percentile the target percentile in range of 0..100
     * @param dimension the dimension to calculate percentile for
     * @return the result {@code NDArray} NDArray
     */
    public NDArray percentile(Number percentile, int[] dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /**
     * Returns median value for this {@code NDArray}.
     *
     * @return the median {@code NDArray}
     */
    public NDArray median() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /**
     * Returns median value along given axes.
     *
     * @param axes the axes along which to perform the median operation
     * @return the median {@code NDArray} along the specified axes
     */
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /**
     * Returns the indices of elements that are non-zero.
     *
     * <p>Note that the behavior is slightly different from numpy.nonzero. Numpy returns a tuple of
     * NDArray, one for each dimension of NDArray. DJL nonzero returns only one {@code NDArray} with
     * last dimension containing all dimension of indices.
     *
     * @return the indices of the elements that are non-zero
     */
    public NDArray nonzero() {
        NDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        return invoke(getParent(), "_npx_nonzero", thisArr, null);
    }

    /**
     * Returns element-wise inverse gauss error function of the {@code NDArray}.
     *
     * @return The inverse of gauss error of the {@code NDArray}, element-wise
     */
    public NDArray erfinv() {
        return invoke(getParent(), "erfinv", this, null);
    }

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * @param keepDims If this is set to True, the axes which are normed over are left in the result
     *     as dimensions with size one. With this option the result will broadcast correctly against
     *     the original x.
     * @return the norm of this {@code NDArray}
     */
    public NDArray norm(boolean keepDims) {
        OpParams params = new OpParams();
        params.add("flag", -2);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_norm", this, params);
    }

    /**
     * Returns the norm of this {@code NDArray}.
     *
     * @param ord Order of the norm.
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     *     the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     *     matrices, and the matrix norms of these matrices are computed.
     * @param keepDims keepDims If this is set to True, the axes which are normed over are left in
     *     the result as dimensions with size one. With this option the result will broadcast
     *     correctly against the original x.
     * @return the norm of this {@code NDArray}
     */
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addParam("ord", (double) ord);
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_norm", this, params);
    }

    //    public MxNDArray oneHot(int depth) {
    //        return LazyNDArray.super.oneHot(depth);
    //    }

    /**
     * Returns a one-hot {@code NDArray}.
     *
     * <ul>
     *   <li>The locations represented by indices take value onValue, while all other locations take
     *       value offValue.
     *   <li>If the input {@code NDArray} is rank N, the output will have rank N+1. The new axis is
     *       appended at the end.
     *   <li>If {@code NDArray} is a scalar the output shape will be a vector of length depth.
     *   <li>If {@code NDArray} is a vector of length features, the output shape will be features x
     *       depth.
     *   <li>If {@code NDArray} is a matrix with shape [batch, features], the output shape will be
     *       batch x features x depth.
     * </ul>
     *
     * @param depth Depth of the one hot dimension.
     * @param onValue The value assigned to the locations represented by indices.
     * @param offValue The value assigned to the locations not represented by indices.
     * @param dataType dataType of the output.
     * @return one-hot encoding of this {@code NDArray}
     */
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        OpParams params = new OpParams();
        params.add("depth", depth);
        params.add("on_value", onValue);
        params.add("off_value", offValue);
        params.add("dtype", dataType);
        return invoke(getParent(), "_npx_one_hot", this, params).toType(dataType, false);
    }

    /**
     * Batchwise product of this {@code NDArray} and the other {@code NDArray}.
     *
     * <ul>
     *   <li>batchDot is used to compute dot product of x and y when x and y are data in batch,
     *       namely N-D (N greater or equal to 3) arrays in shape of (B0, …, B_i, :, :). For
     *       example, given x with shape (B_0, …, B_i, N, M) and y with shape (B_0, …, B_i, M, K),
     *       the result array will have shape (B_0, …, B_i, N, K), which is computed by:
     *       batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ..., b_i, :,
     *       :])
     * </ul>
     *
     * @param other the other {@code NDArray} to perform batch dot product with
     * @return the result {@code NDArray}
     */
    public NDArray batchDot(NDArray other) {
        return invoke(getParent(), "_npx_batch_dot", new NDArray[] {this, other}, null);
    }

    /**
     * Returns an internal representative of Native {@code NDArray}.
     *
     * <p>This method should only be used by Engine provider
     *
     * @return an internal representative of Native {@code NDArray}
     */
    public NDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free NDArray instance: %S", this.getUid()));
            super.freeSubResources();

            if (this.getHandle() != null) {
                JnaUtils.freeNdArray(this.getHandle());
            }
            setClosed(true);
            logger.debug(String.format("Finish to free NDArray instance: %S", this.getUid()));
        }
    }

    /**
     * Returns {@code true} if this {@code NDArray} is special case: no-value {@code NDArray}.
     *
     * @return {@code true} if this NDArray is empty
     */
    public boolean isEmpty() {
        return getShape().size() == 0;
    }

    boolean isSparse() {
        return getSparseFormat() != SparseFormat.DENSE;
    }

    boolean shapeEquals(NDArray other) {
        return getShape().equals(other.getShape());
    }

    /**
     * An engine specific generic invocation to native operation.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not compatible between
     * each version.
     *
     * @param parent the parent {@link MxResource} of the created {@link NDList}
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param params the parameters to be passed to the native operation
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     */
    public static NDList invoke(
            MxResource parent, String operation, NDList src, PairList<String, ?> params) {
        return new NDList(JnaUtils.op(operation).invoke(parent, src.toArray(EMPTY), params));
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     */
    public static void invoke(
            String operation, NDList src, NDList dest, PairList<String, ?> params) {
        invoke(operation, src.toArray(EMPTY), dest.toArray(EMPTY), params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param operation the native operation to perform
     * @param src the array of source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     */
    public static NDArray invoke(
            MxResource parent, String operation, NDArray[] src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(parent, src, params)[0];
    }

    /**
     * An engine specific generic invocation to native operation.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operation
     * @throws IllegalArgumentException if operation is not supported by Engine
     */
    public static void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        JnaUtils.op(operation).invoke(src, dest, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param operation the native operation to perform
     * @param src the source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     */
    public static NDArray invoke(
            MxResource parent, String operation, NDArray src, PairList<String, ?> params) {
        return invoke(parent, operation, new NDArray[] {src}, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param operation the native operation to perform
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     */
    public static NDArray invoke(MxResource parent, String operation, PairList<String, ?> params) {
        return invoke(parent, operation, EMPTY, params);
    }

    /**
     * Encodes {@code MxNDArray} to byte array.
     *
     * @return byte array
     */
    public byte[] encode() {
        return NDSerializer.encode(this);
    }

    /**
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param parent {@link MxResource} of this instance
     * @param low the lower boundary of the output interval. All values generated will be greater
     *     than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    public static NDArray randomUniform(
            MxResource parent,
            float low,
            float high,
            Shape shape,
            DataType dataType,
            Device device) {
        OpParams params = new OpParams();
        params.addParam("low", low);
        params.addParam("high", high);
        params.addParam("size", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke(parent, "_npi_uniform", params);
    }

    /**
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param parent {@link MxResource} of this instance
     * @param low the lower boundary of the output interval. All values generated will be greater
     *     than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    private static NDArray randomUniform(
            MxResource parent, float low, float high, Shape shape, DataType dataType) {
        return randomUniform(parent, low, high, shape, dataType, Device.defaultIfNull(null));
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param parent {@link MxResource} of this instance
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    public static NDArray randomNormal(
            MxResource parent,
            float loc,
            float scale,
            Shape shape,
            DataType dataType,
            Device device) {
        if (device == null) {
            return randomNormal(parent, loc, scale, shape, dataType);
        }
        return randomNormal(parent, loc, scale, shape, dataType);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param parent {@link MxResource} of this instance
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    public static NDArray randomNormal(
            MxResource parent, float loc, float scale, Shape shape, DataType dataType) {
        OpParams params = new OpParams();
        params.addParam("loc", loc);
        params.addParam("scale", scale);
        params.addParam("size", shape);
        params.setDevice(Device.defaultIfNull(null));
        params.setDataType(dataType);
        return invoke(parent, "_npi_normal", params);
    }

    /**
     * Decodes {@link NDArray} through byte array.
     *
     * @param parent the parent {@link MxResource} to create the {@link NDArray}
     * @param bytes byte array to load from
     * @return {@link NDArray}
     */
    static NDArray decode(MxResource parent, byte[] bytes) {
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))) {
            return NDSerializer.decode(parent, dis);
        } catch (IOException e) {
            throw new IllegalArgumentException("NDArray decoding failed", e);
        }
    }

    /**
     * Decodes {@link NDArray} through {@link DataInputStream}.
     *
     * @param parent the parent {@link MxResource} to create the {@link NDArray}
     * @param is input stream data to load from
     * @return {@link NDArray}
     * @throws IOException data is not readable
     */
    public static NDArray decode(MxResource parent, InputStream is) throws IOException {
        return NDSerializer.decode(parent, is);
    }

    /**
     * Converts this {@code NDArray} to a Number array based on its {@link DataType}.
     *
     * @return a Number array
     */
    public Number[] toArray() {
        switch (getDataType()) {
            case FLOAT16:
            case FLOAT32:
                float[] floatArray = toFloatArray();
                return IntStream.range(0, floatArray.length)
                        .mapToObj(i -> floatArray[i])
                        .toArray(Number[]::new);
            case FLOAT64:
                return Arrays.stream(toDoubleArray()).boxed().toArray(Double[]::new);
            case INT32:
                return Arrays.stream(toIntArray()).boxed().toArray(Integer[]::new);
            case INT64:
                return Arrays.stream(toLongArray()).boxed().toArray(Long[]::new);
            case BOOLEAN:
            case INT8:
                ByteBuffer bb = toByteBuffer();
                Byte[] ret = new Byte[bb.remaining()];
                for (int i = 0; i < ret.length; ++i) {
                    ret[i] = bb.get();
                }
                return ret;
            case UINT8:
                return Arrays.stream(toUint8Array()).boxed().toArray(Integer[]::new);
            default:
                throw new IllegalStateException("Unsupported DataType: " + getDataType());
        }
    }

    /**
     * Converts this {@code NDArray} to a boolean array.
     *
     * @return a boolean array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public boolean[] toBooleanArray() {
        if (getDataType() != DataType.BOOLEAN) {
            throw new IllegalStateException(
                    "DataType mismatch, Required boolean" + " Actual " + getDataType());
        }
        ByteBuffer bb = toByteBuffer();
        boolean[] ret = new boolean[bb.remaining()];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = bb.get() != 0;
        }
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a double array.
     *
     * @return a double array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        DoubleBuffer db = toByteBuffer().asDoubleBuffer();
        double[] ret = new double[db.remaining()];
        db.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a float array.
     *
     * @return a float array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public float[] toFloatArray() {
        if (getDataType() == DataType.FLOAT16) {
            return Float16Utils.fromByteBuffer(toByteBuffer());
        } else if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float, Actual " + getDataType());
        }
        FloatBuffer fb = toByteBuffer().asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to an int array.
     *
     * @return an int array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public int[] toIntArray() {
        if (getDataType() != DataType.INT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required int" + " Actual " + getDataType());
        }
        IntBuffer ib = toByteBuffer().asIntBuffer();
        int[] ret = new int[ib.remaining()];
        ib.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a long array.
     *
     * @return a long array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public long[] toLongArray() {
        if (getDataType() != DataType.INT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required long" + " Actual " + getDataType());
        }
        LongBuffer lb = toByteBuffer().asLongBuffer();
        long[] ret = new long[lb.remaining()];
        lb.get(ret);
        return ret;
    }

    /**
     * Converts this {@code NDArray} to a byte array.
     *
     * @return a byte array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public byte[] toByteArray() {
        ByteBuffer bb = toByteBuffer();
        if (bb.hasArray()) {
            return bb.array();
        }
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);
        return buf;
    }

    /**
     * Converts this {@code NDArray} to a uint8 array.
     *
     * @return a uint8 array
     * @throws IllegalStateException when {@link DataType} of this {@code NDArray} mismatches
     */
    public int[] toUint8Array() {
        ByteBuffer bb = toByteBuffer();
        int[] buf = new int[bb.remaining()];
        for (int i = 0; i < buf.length; ++i) {
            buf[i] = bb.get() & 0xff;
        }
        return buf;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (getClosed()) {
            return "This array is already closed";
        }
        return toDebugString(MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /**
     * Runs the debug string representation of this {@code NDArray}.
     *
     * @param maxSize the maximum elements to print out
     * @param maxDepth the maximum depth to print out
     * @param maxRows the maximum rows to print out
     * @param maxColumns the maximum columns to print out
     * @return the debug string representation of this {@code NDArray}
     */
    String toDebugString(int maxSize, int maxDepth, int maxRows, int maxColumns) {
        return NDFormat.format(this, maxSize, maxDepth, maxRows, maxColumns);
    }
}
