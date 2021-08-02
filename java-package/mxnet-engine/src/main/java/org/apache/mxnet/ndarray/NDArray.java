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

public class NDArray extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(NDArray.class);

    protected NDArray(Pointer handle) {
        super(BaseMxResource.getSystemMxResource(), handle);
    }

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

    NDArray(MxResource parent, Pointer handle) {
        super(parent, handle);
        this.mxNDArrayEx = new NDArrayEx(this);
    }

    NDArray(MxResource parent, Pointer handle, SparseFormat fmt) {
        this(parent, handle);
        this.sparseFormat = fmt;
    }

    public static NDArray create(MxResource parent, Pointer handle) {
        return new NDArray(parent, handle);
    }

    public static NDArray create(MxResource parent, Pointer handle, SparseFormat fmt) {
        return new NDArray(parent, handle, fmt);
    }

    public static NDArray create(MxResource parent, Shape shape, Device device) {
        return create(parent, shape, DataType.FLOAT32, device);
    }

    public static NDArray create(MxResource parent, Shape shape) {
        return create(parent, shape, DataType.FLOAT32, Device.defaultIfNull());
    }

    public static NDArray create(
            MxResource parent, Shape shape, DataType dataType, Device device, boolean hasGradient) {
        Pointer handle =
                JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), hasGradient);
        return new NDArray(parent, handle, device, shape, dataType, hasGradient);
    }

    public static NDArray create(MxResource parent, Shape shape, DataType dataType, Device device) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        return new NDArray(parent, handle, Device.defaultIfNull(device), shape, dataType, false);
    }

    public static NDArray ones(MxResource parent, Shape shape, DataType dataType, Device device) {
        return create(parent, shape, dataType, device).ones();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public DataType getDataType() {
        if (this.dataType == null) {
            this.dataType = JnaUtils.getDataTypeOfNdArray(getHandle());
        }
        return this.dataType;
    }

    public Device getDevice() {
        if (this.device == null) {
            this.device = JnaUtils.getDeviceOfNdArray(getHandle());
        }
        return this.device;
    }

    public Shape getShape() {
        if (this.shape == null) {
            this.shape = JnaUtils.getShapeOfNdArray(getHandle());
        }
        return this.shape;
    }

    public SparseFormat getSparseFormat() {
        if (this.sparseFormat == null) {
            this.sparseFormat = JnaUtils.getStorageType(getHandle());
        }
        return this.sparseFormat;
    }

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

    NDArray duplicate() {
        NDArray array = create(getParent(), getShape(), getDataType(), getDevice());
        array.setName(getName());
        copyTo(array);
        return array;
    }

    public NDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice()) && !copy) {
            return this;
        }
        return duplicate(getShape(), getDataType(), device, getName());
    }

    public NDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        return duplicate(getShape(), dataType, getDevice(), getName());
    }

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

    public NDArray zeros(Shape shape, DataType dataType) {
        return fill("_npi_zeros", shape, dataType);
    }

    public NDArray zeros() {
        return fill("_npi_zeros", getShape(), getDataType());
    }

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
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray ones(Shape shape, DataType dataType) {
        return fill("_npi_ones", shape, dataType);
    }

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

    public NDArray getGradient() {
        if (!hasGradient()) {
            throw new IllegalStateException(
                    "No gradient attached to this MxNDArray, please call array.requiredGradient()"
                            + "on your MxNDArray or block.setInitializer() on your Block");
        }
        Pointer pointer = JnaUtils.getGradient(getHandle());
        return create(getParent(), pointer);
    }

    public boolean hasGradient() {
        if (hasGradient == null) {
            Pointer pointer = JnaUtils.getGradient(getHandle());
            hasGradient = pointer != null;
        }
        return hasGradient;
    }

    public NDArray stopGradient() {
        Pointer pointer = JnaUtils.detachGradient(getHandle());
        return create(getParent(), pointer);
    }

    public String[] toStringArray() {
        throw new UnsupportedOperationException("String MxNDArray is not supported!");
    }

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

    NDArray all() {
        // result of sum operation is int64 now
        return toType(DataType.BOOLEAN, false).sum().eq(size());
    }

    public void copyTo(NDArray NdArray) {

        Shape inShape = getShape();
        Shape destShape = NdArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        JnaUtils.op("_npi_copyto").invoke(new NDArray[] {this}, new NDArray[] {NdArray}, null);
    }

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

    public NDArray sequenceMask(NDArray sequenceLength) {
        return sequenceMask(sequenceLength, 0);
    }

    public NDArray zerosLike() {
        OpParams params = new OpParams();
        params.addParam("fill_value", 0);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    public NDArray onesLike() {
        OpParams params = new OpParams();
        params.addParam("fill_value", 1);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    NDArray get(NDIndex index) {
        return getNDArrayInternal().getIndexer().get(this, index);
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

    public boolean contentEquals(Number number) {
        if (number == null) {
            return false;
        }
        try (NDArray result = eq(number)) {
            return result.all().getBoolean();
        }
    }

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

    public NDArray eq(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_equal_scalar", this, params);
    }

    public NDArray eq(NDArray other) {
        return invoke(getParent(), "_npi_equal", new NDArray[] {this, other}, null);
    }

    public NDArray neq(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_not_equal_scalar", this, params);
    }

    public NDArray neq(NDArray other) {
        return invoke(getParent(), "_npi_not_equal", new NDArray[] {this, other}, null);
    }

    public NDArray gt(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_scalar", this, params);
    }

    public NDArray gt(NDArray other) {
        return invoke(getParent(), "_npi_greater", new NDArray[] {this, other}, null);
    }

    public NDArray gte(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_equal_scalar", this, params);
    }

    public NDArray gte(NDArray other) {
        return invoke(getParent(), "_npi_greater_equal", new NDArray[] {this, other}, null);
    }

    public NDArray lt(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_scalar", this, params);
    }

    public NDArray lt(NDArray other) {
        return invoke(getParent(), "_npi_less", new NDArray[] {this, other}, null);
    }

    public NDArray lte(Number other) {
        OpParams params = new OpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_equal_scalar", this, params);
    }

    public NDArray lte(NDArray other) {
        return invoke(getParent(), "_npi_less_equal", new NDArray[] {this, other}, null);
    }

    public NDArray add(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_add_scalar", this, params);
    }

    public NDArray add(NDArray other) {
        return invoke(getParent(), "_npi_add", new NDArray[] {this, other}, null);
    }

    public NDArray sub(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_subtract_scalar", this, params);
    }

    public NDArray sub(NDArray other) {
        return invoke(getParent(), "_npi_subtract", new NDArray[] {this, other}, null);
    }

    public NDArray mul(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_multiply_scalar", this, params);
    }

    public NDArray mul(NDArray other) {
        return invoke(getParent(), "_npi_multiply", new NDArray[] {this, other}, null);
    }

    public NDArray div(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_true_divide_scalar", this, params);
    }

    public NDArray div(NDArray other) {
        return invoke(getParent(), "_npi_true_divide", new NDArray[] {this, other}, null);
    }

    public NDArray mod(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_mod_scalar", this, params);
    }

    public NDArray mod(NDArray other) {
        return invoke(getParent(), "_npi_mod", new NDArray[] {this, other}, null);
    }

    public NDArray pow(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_power_scalar", this, params);
    }

    public NDArray pow(NDArray other) {
        return invoke(getParent(), "_npi_power", new NDArray[] {this, other}, null);
    }

    public NDArray addi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_add_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray addi(NDArray other) {
        invoke("_npi_add", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray subi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_subtract_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray subi(NDArray other) {
        invoke("_npi_subtract", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray muli(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_multiply_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray muli(NDArray other) {
        invoke("_npi_multiply", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray divi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_true_divide_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray divi(NDArray other) {
        invoke("_npi_true_divide", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray modi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_mod_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray modi(NDArray other) {
        invoke("_npi_mod", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray powi(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        invoke("_npi_power_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    public NDArray powi(NDArray other) {
        invoke("_npi_power", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray sign() {
        return invoke(getParent(), "_npi_sign", this, null);
    }

    public NDArray signi() {
        invoke("_npi_sign", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray maximum(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_maximum_scalar", this, params);
    }

    public NDArray maximum(NDArray other) {
        return invoke(getParent(), "_npi_maximum", new NDArray[] {this, other}, null);
    }

    public NDArray minimum(Number n) {
        OpParams params = new OpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_minimum_scalar", this, params);
    }

    public NDArray minimum(NDArray other) {
        return invoke(getParent(), "_npi_minimum", new NDArray[] {this, other}, null);
    }

    public NDArray neg() {
        return invoke(getParent(), "_npi_negative", this, null);
    }

    public NDArray negi() {
        invoke("_npi_negative", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    public NDArray abs() {
        return invoke(getParent(), "_npi_absolute", this, null);
    }

    public NDArray square() {
        return invoke(getParent(), "_npi_square", this, null);
    }

    public NDArray sqrt() {
        return invoke(getParent(), "_npi_sqrt", this, null);
    }

    public NDArray cbrt() {
        return invoke(getParent(), "_npi_cbrt", this, null);
    }

    public NDArray floor() {
        return invoke(getParent(), "_npi_floor", this, null);
    }

    public NDArray ceil() {
        return invoke(getParent(), "_npi_ceil", this, null);
    }

    public NDArray round() {
        return invoke(getParent(), "round", this, null);
    }

    public NDArray trunc() {
        return invoke(getParent(), "_npi_trunc", this, null);
    }

    public NDArray exp() {
        return invoke(getParent(), "_npi_exp", this, null);
    }

    public NDArray log() {
        return invoke(getParent(), "_npi_log", this, null);
    }

    public NDArray log10() {
        return invoke(getParent(), "_npi_log10", this, null);
    }

    public NDArray log2() {
        return invoke(getParent(), "_npi_log2", this, null);
    }

    public NDArray sin() {
        return invoke(getParent(), "_npi_sin", this, null);
    }

    public NDArray cos() {
        return invoke(getParent(), "_npi_cos", this, null);
    }

    public NDArray tan() {
        return invoke(getParent(), "_npi_tan", this, null);
    }

    public NDArray asin() {
        return invoke(getParent(), "_npi_arcsin", this, null);
    }

    public NDArray acos() {
        return invoke(getParent(), "_npi_arccos", this, null);
    }

    public NDArray atan() {
        return invoke(getParent(), "_npi_arctan", this, null);
    }

    public NDArray sinh() {
        return invoke(getParent(), "_npi_sinh", this, null);
    }

    public NDArray cosh() {
        return invoke(getParent(), "_npi_cosh", this, null);
    }

    public NDArray tanh() {
        return invoke(getParent(), "_npi_tanh", this, null);
    }

    public NDArray asinh() {
        return invoke(getParent(), "_npi_arcsinh", this, null);
    }

    public NDArray acosh() {
        return invoke(getParent(), "_npi_arccosh", this, null);
    }

    public NDArray atanh() {
        return invoke(getParent(), "_npi_arctanh", this, null);
    }

    public NDArray toDegrees() {
        return invoke(getParent(), "_npi_degrees", this, null);
    }

    public NDArray toRadians() {
        return invoke(getParent(), "_npi_radians", this, null);
    }

    public NDArray max() {
        return invoke(getParent(), "_np_max", this, null);
    }

    public NDArray max(int[] axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_max", this, params);
    }

    public NDArray max(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_max", this, params);
    }

    public NDArray min() {
        return invoke(getParent(), "_np_min", this, null);
    }

    public NDArray min(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_min", this, params);
    }

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

    public NDArray sum(int[] axes) {
        return sum(axes, false);
    }

    public NDArray sum(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_sum", this, params);
    }

    public NDArray prod() {
        return invoke(getParent(), "_np_prod", this, null);
    }

    NDArray prod(int[] axes) {
        return prod(axes, false);
    }

    public NDArray prod(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_prod", this, params);
    }

    public NDArray mean() {
        return invoke(getParent(), "_npi_mean", this, null);
    }

    public NDArray mean(int[] axes, boolean keepDims) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_mean", this, params);
    }

    public NDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        OpParams params = new OpParams();
        params.addTupleParam("axes", axes);
        params.addParam("k", times);
        return invoke(getParent(), "_npi_rot90", this, params);
    }

    public NDArray trace(int offset, int axis1, int axis2) {
        OpParams params = new OpParams();
        params.addParam("offset", offset);
        params.addParam("axis1", axis1);
        params.addParam("axis2", axis2);
        return invoke(getParent(), "_np_trace", this, params);
    }

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

    public NDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    public NDArray reshape(Shape shape) {
        OpParams params = new OpParams();
        params.addParam("newshape", shape);
        return invoke(getParent(), "_np_reshape", this, params);
    }

    public NDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    public NDArray expandDims(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_expand_dims", this, params);
    }

    public NDArray squeeze() {
        return invoke(getParent(), "_np_squeeze", this, null);
    }

    public NDArray squeeze(int[] axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_squeeze", this, params);
    }

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

    public NDArray logicalNot() {
        return invoke(getParent(), "_npi_logical_not", this, null);
    }

    public NDArray argSort(int axis, boolean ascending) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT64);
        return invoke(getParent(), "_npi_argsort", this, params);
    }

    public NDArray sort(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_sort", this, params);
    }

    public NDArray sort() {
        return invoke(getParent(), "_npi_sort", this, null);
    }

    public NDArray softmax(int axis) {
        // MXNet softmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_softmax", this, params);
    }

    public NDArray logSoftmax(int axis) {
        // MXNet logsoftmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_log_softmax", this, params);
    }

    public NDArray cumSum() {
        return invoke(getParent(), "_np_cumsum", this, null);
    }

    public NDArray cumSum(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_np_cumsum", this, params);
    }

    public void intern(NDArray replaced) {
        NDArray arr = replaced;
        Pointer oldHandle = handle.getAndSet(arr.handle.getAndSet(null));
        JnaUtils.waitToRead(oldHandle);
        JnaUtils.freeNdArray(oldHandle);
        // dereference old ndarray
        arr.close();
    }

    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    public NDArray isNaN() {
        return invoke(getParent(), "_npi_isnan", this, null);
    }

    public NDArray toDense() {
        if (!isSparse()) {
            return duplicate();
        }
        return castStorage(SparseFormat.DENSE);
    }

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

    private int withAxis(int axis) {
        return Math.floorMod(axis, getShape().dimension());
    }

    public NDArray tile(long[] repeats) {
        OpParams params = new OpParams();
        params.addTupleParam("reps", repeats);
        return invoke(getParent(), "_npi_tile", this, params);
    }

    public NDArray tile(Shape desiredShape) {
        return tile(repeatsToMatchShape(desiredShape));
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

    public NDArray repeat(int axis, long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return repeat(repeatsArray);
    }

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

    public NDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    public NDArray dot(NDArray other) {
        return invoke(getParent(), "_np_dot", new NDArray[] {this, other}, null);
    }

    public NDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        return invoke(getParent(), "_npi_matmul", new NDArray[] {this, other}, null);
    }

    public NDArray clip(Number min, Number max) {
        OpParams params = new OpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return invoke(getParent(), "_npi_clip", this, params);
    }

    public NDArray swapAxes(int axis1, int axis2) {
        OpParams params = new OpParams();
        params.addParam("dim1", axis1);
        params.addParam("dim2", axis2);
        return invoke(getParent(), "_npi_swapaxes", this, params);
    }

    public NDArray flip(int... axes) {
        OpParams params = new OpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_npi_flip", this, params);
    }

    public NDArray transpose() {
        return invoke(getParent(), "_np_transpose", this, null);
    }

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

    public NDArray broadcast(Shape shape) {
        OpParams params = new OpParams();
        params.setShape(shape);
        return invoke(getParent(), "_npi_broadcast_to", this, params);
    }

    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmax", this, null);
    }

    public NDArray argMax(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmax", this, params);
    }

    public NDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmin", this, null);
    }

    public NDArray argMin(int axis) {
        OpParams params = new OpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmin", this, params);
    }

    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    public NDArray percentile(Number percentile, int[] dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    public NDArray median() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    public NDArray nonzero() {
        NDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        return invoke(getParent(), "_npx_nonzero", thisArr, null);
    }

    public NDArray erfinv() {
        return invoke(getParent(), "erfinv", this, null);
    }

    public NDArray norm(boolean keepDims) {
        OpParams params = new OpParams();
        params.add("flag", -2);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_norm", this, params);
    }

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

    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        OpParams params = new OpParams();
        params.add("depth", depth);
        params.add("on_value", onValue);
        params.add("off_value", offValue);
        params.add("dtype", dataType);
        return invoke(getParent(), "_npx_one_hot", this, params).toType(dataType, false);
    }

    public NDArray batchDot(NDArray other) {
        return invoke(getParent(), "_npx_batch_dot", new NDArray[] {this, other}, null);
    }

    public NDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free NDArray instance: %S", this.getUid()));
            super.freeSubResources();

            if (this.getHandle() != null) {
                JnaUtils.freeNdArray(this.getHandle());
            }
            setClosed();
            logger.debug(String.format("Finish to free NDArray instance: %S", this.getUid()));
        }
    }

    public boolean isEmpty() {
        return getShape().size() == 0;
    }

    boolean isSparse() {
        return getSparseFormat() != SparseFormat.DENSE;
    }

    NDArray booleanMask(NDArray index) {
        return booleanMask(index, 0);
    }

    boolean shapeEquals(NDArray other) {
        return getShape().equals(other.getShape());
    }

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
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
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
}
