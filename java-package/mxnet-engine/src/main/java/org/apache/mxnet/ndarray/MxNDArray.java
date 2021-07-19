package org.apache.mxnet.ndarray;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.GradReq;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;
import org.apache.mxnet.util.Float16Utils;
import org.apache.mxnet.util.PairList;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

public class MxNDArray extends MxResource {

    protected MxNDArray(Pointer handle) {
        super(BaseMxResource.getSystemMxResource(), handle);
    }

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;
    private static final MxNDArray[] EMPTY = new MxNDArray[0];

    private String name;
    private Device device;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    // use Boolean object to maintain three status: false, true
    // and null which means the flag is not set by the native engine yet
    private Boolean hasGradient;
    private Integer version;
    private MxNDArrayEx mxNDArrayEx;

    MxNDArray(
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

    MxNDArray(MxResource parent, Pointer handle) {
        super(parent, handle);
        this.mxNDArrayEx = new MxNDArrayEx(this);
    }

    MxNDArray(MxResource parent, Pointer handle, SparseFormat fmt) {
        this(parent, handle);
        this.sparseFormat = fmt;
    }

    public static MxNDArray create(MxResource parent, Pointer handle) {
        return new MxNDArray(parent, handle);
    }

    public static MxNDArray create(MxResource parent, Pointer handle, SparseFormat fmt) {
        return new MxNDArray(parent, handle, fmt);
    }

    public static MxNDArray create(MxResource parent, Shape shape, Device device) {
        return create(parent, shape, DataType.FLOAT32 ,device);
    }

    public static MxNDArray create(MxResource parent, Shape shape, DataType dataType, Device device, boolean hasGradient) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), hasGradient);
        return new MxNDArray(parent, handle, device, shape, dataType, hasGradient);

    }
    public static MxNDArray create(MxResource parent, Shape shape, DataType dataType, Device device) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        return new MxNDArray(parent, handle, Device.defaultIfNull(device), shape, dataType, false);
    }


    public static MxNDArray ones(MxResource parent, Shape shape, DataType dataType, Device device) {
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


    private MxNDArray duplicate(
            Shape shape, DataType dataType, Device device, String name
    ) {
        // TODO get copy parameter
        MxNDArray array = create(getParent(), shape, dataType, device);
        array.setName(name);
        copyTo(array);
        return array;
    }

    MxNDArray duplicate() {
        MxNDArray array = create(getParent(), getShape(), getDataType(), getDevice());
        array.setName(getName());
        copyTo(array);
        return array;
    }

    
    public MxNDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice()) && !copy) {
            return this;
        }
        return duplicate(getShape(), getDataType(), device, getName());
    }

    
    public MxNDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        return duplicate(getShape(), dataType, getDevice(), getName());
    }

    
    public void setRequiresGradient(boolean requiresGrad) {
        if ((requiresGrad && hasGradient()) || (!requiresGrad && !hasGradient())) {
            return;
        }
        MxNDArray grad =
                hasGradient() ? getGradient() : createGradient(getSparseFormat());
        // DJL go with write as only MXNet support GradReq
        int gradReqValue = requiresGrad ? GradReq.WRITE.getValue() : GradReq.NULL.getValue();
        IntBuffer gradReqBuffer = IntBuffer.allocate(1);
        gradReqBuffer.put(0, gradReqValue);
        JnaUtils.autogradMarkVariables(1, getHandle(), gradReqBuffer, grad.getHandle());
        hasGradient = requiresGrad;
        grad.close();
    }

    public MxNDArray zeros(Shape shape, DataType dataType) {
        return fill("_npi_zeros", shape, dataType);
    }

    public MxNDArray zeros() {
        return fill("_npi_zeros", getShape(), getDataType());
    }

    MxNDArray zeros(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return zeros(shape, dataType);
        }
        return zeros(shape, dataType);
    }

    private MxNDArray createGradient(SparseFormat format) {
        try (MxNDArray zeros = this.zeros(getShape(), getDataType(), getDevice())) {
            return zeros.toSparse(format);
        }
    }

    private MxNDArray fill(String opName, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        if (shape == null) {
            throw new IllegalArgumentException("Shape is required for " + opName.substring(1));
        }
        params.addParam("shape", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke(getParent(), opName, params);
    }

    /**
     * Creates an instance of {@link MxNDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @param dataType the {@link DataType} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    MxNDArray ones(Shape shape, DataType dataType) {
        return fill("_npi_ones", shape, dataType);
    }

    public MxNDArray ones() {
        return ones(getShape(), getDataType());
    }
    /**
     * Creates an instance of {@link MxNDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    MxNDArray ones(Shape shape) {
        return ones(shape, DataType.FLOAT32);
    }

    /**
     * Creates an instance of {@link MxNDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @param dataType the {@link DataType} of the {@link MxNDArray}
     * @param device the {@link Device} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    MxNDArray ones(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return ones(shape, dataType);
        }
        return create(getParent(), shape, dataType, device).ones();
    }

    
    public MxNDArray getGradient() {
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

    
    public MxNDArray stopGradient() {
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
        ByteBuffer bb = MxNDSerializer.allocateDirect(Math.toIntExact(len));
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
        ByteBuffer buf = MxNDSerializer.allocateDirect(size * numOfBytes);

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
     * Returns {@code true} if this {@code MxNDArray} is a scalar {@code MxNDArray} with empty {@link
     * Shape}.
     *
     * @return {@code true} if this {@code MxNDArray} is a scalar {@code MxNDArray} with empty {@link
     *     Shape}
     */
    boolean isScalar() {
        return getShape().isScalar();
    }

    MxNDArray all() {
        // result of sum operation is int64 now
        return toType(DataType.BOOLEAN, false).sum().eq(size());
    }
    
    public void copyTo(MxNDArray mxNdArray) {

        Shape inShape = getShape();
        Shape destShape = mxNdArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        JnaUtils.op("_npi_copyto").invoke(new MxNDArray[] {this}, new MxNDArray[] {mxNdArray}, null);
    }

    
    public MxNDArray booleanMask(MxNDArray index, int axis) {
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
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        try (MxNDArray reshaped = this.reshape(new Shape(reshape));
             MxNDArray reshapedIndex = index.toType(DataType.INT32, false).reshape(-1);
             MxNDArray result =
                     invoke(
                             getParent(),
                             "_npi_boolean_mask",
                             new MxNDArray[] {reshaped, reshapedIndex},
                             params)) {
            return result.reshape(reshape);
        }
    }
    
    public MxNDArray sequenceMask(MxNDArray sequenceLength, float value) {
        if (getShape().dimension() < 2 || getShape().isScalar() || getShape().hasZeroDimension()) {
            throw new IllegalArgumentException(
                    "sequenceMask is not supported for MxNDArray with less than 2 dimensions");
        }
        Shape expectedSequenceLengthShape = new Shape(getShape().get(0));
        if (!sequenceLength.getShape().equals(expectedSequenceLengthShape)) {
            throw new IllegalArgumentException("SequenceLength must be of shape [batchSize]");
        }
        MxOpParams params = new MxOpParams();
        params.add("value", value);
        params.add("use_sequence_length", true);
        params.add("axis", 1);
        return invoke(getParent(), "_npx_sequence_mask", new MxNDList(this, sequenceLength), params)
                .head();
    }

    
    public MxNDArray sequenceMask(MxNDArray sequenceLength) {
        return sequenceMask(sequenceLength, 0);
    }

    
    public MxNDArray zerosLike() {
        MxOpParams params = new MxOpParams();
        params.addParam("fill_value", 0);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    
    public MxNDArray onesLike() {
        MxOpParams params = new MxOpParams();
        params.addParam("fill_value", 1);
        return invoke(getParent(), "_npi_full_like", this, params);
    }

    MxNDArray get(NDIndex index) {
        return getNDArrayInternal().getIndexer().get(this, index);
    }

    MxNDArray getScalar(long... indices) {
        MxNDArray value = get(new NDIndex(indices));
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
        try (MxNDArray result = eq(number)) {
            return result.all().getBoolean();
        }
    }

    
    public boolean contentEquals(MxNDArray other) {
        if (other == null || (!shapeEquals(other))) {
            return false;
        }
        if (getDataType() != other.getDataType()) {
            return false;
        }
        try (MxNDArray result = eq(other).toType(DataType.INT32, false)) {
            return result.all().getBoolean();
        }
    }

    
    public MxNDArray eq(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_equal_scalar", this, params);
    }

    
    public MxNDArray eq(MxNDArray other) {
        return invoke(getParent(), "_npi_equal", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray neq(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_not_equal_scalar", this, params);
    }

    
    public MxNDArray neq(MxNDArray other) {
        return invoke(getParent(), "_npi_not_equal", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray gt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_scalar", this, params);
    }

    
    public MxNDArray gt(MxNDArray other) {
        return invoke(getParent(), "_npi_greater", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray gte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_greater_equal_scalar", this, params);
    }

    
    public MxNDArray gte(MxNDArray other) {
        return invoke(getParent(), "_npi_greater_equal", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray lt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_scalar", this, params);
    }

    
    public MxNDArray lt(MxNDArray other) {
        return invoke(getParent(), "_npi_less", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray lte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return invoke(getParent(), "_npi_less_equal_scalar", this, params);
    }

    
    public MxNDArray lte (MxNDArray other) {
        return invoke(getParent(), "_npi_less_equal", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray add(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_add_scalar", this, params);
    }

    
    public MxNDArray add (MxNDArray other) {
        return invoke(getParent(), "_npi_add", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray sub(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_subtract_scalar", this, params);
    }

    
    public MxNDArray sub (MxNDArray other) {
        return invoke(getParent(), "_npi_subtract", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray mul(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_multiply_scalar", this, params);
    }

    
    public MxNDArray mul (MxNDArray other) {
        return invoke(getParent(), "_npi_multiply", new MxNDArray[] {this, other}, null);
    }


    
    public MxNDArray div(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_true_divide_scalar", this, params);
    }

    
    public MxNDArray div (MxNDArray other) {
        return invoke(getParent(), "_npi_true_divide", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray mod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_mod_scalar", this, params);
    }

    
    public MxNDArray mod (MxNDArray other) {
        return invoke(getParent(), "_npi_mod", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray pow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_power_scalar", this, params);
    }

    
    public MxNDArray pow (MxNDArray other) {
        return invoke(getParent(), "_npi_power", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray addi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke("_npi_add_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray addi (MxNDArray other) {
        invoke("_npi_add", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray subi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke("_npi_subtract_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray subi (MxNDArray other) {
        invoke("_npi_subtract", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray muli(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke("_npi_multiply_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray muli (MxNDArray other) {
        invoke("_npi_multiply", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray divi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke(
                "_npi_true_divide_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray divi (MxNDArray other) {
        invoke("_npi_true_divide", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray modi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke("_npi_mod_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray modi (MxNDArray other) {
        invoke("_npi_mod", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray powi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        invoke("_npi_power_scalar", new MxNDArray[] {this}, new MxNDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray powi (MxNDArray other) {
        invoke("_npi_power", new MxNDArray[] {this, other}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray sign() {
        return invoke(getParent(), "_npi_sign", this, null);
    }

    
    public MxNDArray signi() {
        invoke("_npi_sign", new MxNDArray[] {this}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray maximum(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_maximum_scalar", this, params);
    }

    
    public MxNDArray maximum (MxNDArray other) {
        return invoke(getParent(), "_npi_maximum", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray minimum(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return invoke(getParent(), "_npi_minimum_scalar", this, params);
    }

    
    public MxNDArray minimum (MxNDArray other) {
        return invoke(getParent(), "_npi_minimum", new MxNDArray[] {this, other}, null);
    }
    
    public MxNDArray neg() {
        return invoke(getParent(), "_npi_negative", this, null);
    }

    
    public MxNDArray negi() {
        invoke("_npi_negative", new MxNDArray[] {this}, new MxNDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray abs() {
        return invoke(getParent(), "_npi_absolute", this, null);
    }

    
    public MxNDArray square() {
        return invoke(getParent(), "_npi_square", this, null);
    }

    
    public MxNDArray sqrt() {
        return invoke(getParent(), "_npi_sqrt", this, null);
    }

    
    public MxNDArray cbrt() {
        return invoke(getParent(), "_npi_cbrt", this, null);
    }

    
    public MxNDArray floor() {
        return invoke(getParent(), "_npi_floor", this, null);
    }

    
    public MxNDArray ceil() {
        return invoke(getParent(), "_npi_ceil", this, null);
    }

    
    public MxNDArray round() {
        return invoke(getParent(), "round", this, null);
    }

    
    public MxNDArray trunc() {
        return invoke(getParent(), "_npi_trunc", this, null);
    }

    
    public MxNDArray exp() {
        return invoke(getParent(), "_npi_exp", this, null);
    }

    
    public MxNDArray log() {
        return invoke(getParent(), "_npi_log", this, null);
    }

    
    public MxNDArray log10() {
        return invoke(getParent(), "_npi_log10", this, null);
    }

    
    public MxNDArray log2() {
        return invoke(getParent(), "_npi_log2", this, null);
    }

    
    public MxNDArray sin() {
        return invoke(getParent(), "_npi_sin", this, null);
    }

    
    public MxNDArray cos() {
        return invoke(getParent(), "_npi_cos", this, null);
    }

    
    public MxNDArray tan() {
        return invoke(getParent(), "_npi_tan", this, null);
    }

    
    public MxNDArray asin() {
        return invoke(getParent(), "_npi_arcsin", this, null);
    }

    
    public MxNDArray acos() {
        return invoke(getParent(), "_npi_arccos", this, null);
    }

    
    public MxNDArray atan() {
        return invoke(getParent(), "_npi_arctan", this, null);
    }

    
    public MxNDArray sinh() {
        return invoke(getParent(), "_npi_sinh", this, null);
    }

    
    public MxNDArray cosh() {
        return invoke(getParent(), "_npi_cosh", this, null);
    }

    
    public MxNDArray tanh() {
        return invoke(getParent(), "_npi_tanh", this, null);
    }

    
    public MxNDArray asinh() {
        return invoke(getParent(), "_npi_arcsinh", this, null);
    }

    
    public MxNDArray acosh() {
        return invoke(getParent(), "_npi_arccosh", this, null);
    }

    
    public MxNDArray atanh() {
        return invoke(getParent(), "_npi_arctanh", this, null);
    }

    
    public MxNDArray toDegrees() {
        return invoke(getParent(), "_npi_degrees", this, null);
    }

    
    public MxNDArray toRadians() {
        return invoke(getParent(), "_npi_radians", this, null);
    }


    
    public MxNDArray max() {
        return invoke(getParent(), "_np_max", this, null);
    }

    
    public MxNDArray max(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_max", this, params);
    }

    
    public MxNDArray max(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_max", this, params);
    }

    
    public MxNDArray min() {
        return invoke(getParent(), "_np_min", this, null);
    }

    
    public MxNDArray min(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_min", this, params);
    }

    
    public MxNDArray sum() {
        // TODO current windows doesn't support boolean MxNDArray
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            DataType target = getDataType();
            if (!target.isFloating()) {
                try  (MxNDArray thisArr = toType(DataType.FLOAT32, false)) {
                    if (target == DataType.BOOLEAN) {
                        target = DataType.INT64;
                    }
                    try  (MxNDArray array = invoke(getParent(), "_np_sum", thisArr, null)) {
                        return array.toType(target, false);
                    }
                }
            }
        }
        return invoke(getParent(), "_np_sum", this, null);
    }

    public MxNDArray sum(int[] axes) {
        return sum(axes, false);
    }

    
    public MxNDArray sum(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_sum", this, params);
    }

    
    public MxNDArray prod() {
        return invoke(getParent(), "_np_prod", this, null);
    }

    MxNDArray prod(int[] axes) {
        return prod(axes, false);
    }


    public MxNDArray prod(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_np_prod", this, params);
    }

    
    public MxNDArray mean() {
        return invoke(getParent(), "_npi_mean", this, null);
    }

    
    public MxNDArray mean(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_mean", this, params);
    }

    
    public MxNDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axes", axes);
        params.addParam("k", times);
        return invoke(getParent(), "_npi_rot90", this, params);
    }

    
    public MxNDArray trace(int offset, int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("offset", offset);
        params.addParam("axis1", axis1);
        params.addParam("axis2", axis2);
        return invoke(getParent(), "_np_trace", this, params);
    }

    
    public MxNDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new MxNDList(this);
        }
        MxOpParams params = new MxOpParams();
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
        return invoke(getParent(), "_npi_split", new MxNDList(this), params);
    }

    
    public MxNDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    
    public MxNDArray reshape(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.addParam("newshape", shape);
        return invoke(getParent(), "_np_reshape", this, params);
    }

    public MxNDArray reshape(long... newShape) {
        return reshape(new Shape(newShape));
    }

    
    public MxNDArray expandDims(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_expand_dims", this, params);
    }

    
    public MxNDArray squeeze() {
        return invoke(getParent(), "_np_squeeze", this, null);
    }

    
    public MxNDArray squeeze(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_np_squeeze", this, params);
    }

    
    public MxNDArray logicalAnd (MxNDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_and", new MxNDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalOr (MxNDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_or", new MxNDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalXor (MxNDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        other =
                (other.getDataType() == DataType.BOOLEAN)
                        ? other.toType(DataType.INT32, false)
                        : other;
        return invoke(getParent(), "broadcast_logical_xor", new MxNDArray[] {thisArr, other}, null)
                .toType(DataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalNot() {
        return invoke(getParent(), "_npi_logical_not", this, null);
    }

    
    public MxNDArray argSort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT64);
        return invoke(getParent(), "_npi_argsort", this, params);
    }

    
    public MxNDArray sort(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_sort", this, params);
    }

    
    public MxNDArray sort() {
        return invoke(getParent(), "_npi_sort", this, null);
    }

    
    public MxNDArray softmax(int axis) {
        // MXNet softmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_softmax", this, params);
    }

    
    public MxNDArray logSoftmax(int axis) {
        // MXNet logsoftmax op bug on GPU
        if (isEmpty()) {
            return create(getParent(), getShape(), DataType.FLOAT32, getDevice());
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npx_log_softmax", this, params);
    }

    
    public MxNDArray cumSum() {
        return invoke(getParent(), "_np_cumsum", this, null);
    }

    
    public MxNDArray cumSum(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_np_cumsum", this, params);
    }

    
    public void intern (MxNDArray replaced) {
        MxNDArray arr = replaced;
        Pointer oldHandle = handle.getAndSet(arr.handle.getAndSet(null));
        JnaUtils.waitToRead(oldHandle);
        JnaUtils.freeNdArray(oldHandle);
        // dereference old ndarray
        arr.close();
    }

    
    public MxNDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    
    public MxNDArray isNaN() {
        return invoke(getParent(), "_npi_isnan", this, null);
    }

    
    public MxNDArray toDense() {
        if (!isSparse()) {
            return duplicate();
        }
        return castStorage(SparseFormat.DENSE);
    }

    
    public MxNDArray toSparse(SparseFormat fmt) {
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

    private MxNDArray castStorage(SparseFormat fmt) {
        MxOpParams params = new MxOpParams();
        params.setParam("stype", fmt.getType());
        return invoke(getParent(), "cast_storage", this, params);
    }
    
    public MxNDArray tile(long repeats) {
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

    
    public MxNDArray tile(int axis, long repeats) {
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
    
    public MxNDArray tile(long[] repeats) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("reps", repeats);
        return invoke(getParent(), "_npi_tile", this, params);
    }

    
    public MxNDArray tile(Shape desiredShape) {
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

    
    public MxNDArray repeat(long repeats) {
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

    
    public MxNDArray repeat(int axis, long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return repeat(repeatsArray);
    }

    
    public MxNDArray repeat(long[] repeats) {
        // TODO get rid of for loop once bug in MXNet np.repeat is fixed
        MxNDArray array = this;
        int baseAxis = getShape().dimension() - repeats.length;
        for (int i = 0; i < repeats.length; i++) {
            if (repeats[i] > 1) {
                MxNDArray previousArray = array;
                MxOpParams params = new MxOpParams();
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

    
    public MxNDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    
    public MxNDArray dot (MxNDArray other) {
        return invoke(getParent(), "_np_dot", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray matMul (MxNDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        return invoke(getParent(), "_npi_matmul", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArray clip(Number min, Number max) {
        MxOpParams params = new MxOpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return invoke(getParent(), "_npi_clip", this, params);
    }

    
    public MxNDArray swapAxes(int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("dim1", axis1);
        params.addParam("dim2", axis2);
        return invoke(getParent(), "_npi_swapaxes", this, params);
    }

    
    public MxNDArray flip(int... axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return invoke(getParent(), "_npi_flip", this, params);
    }

    
    public MxNDArray transpose() {
        return invoke(getParent(), "_np_transpose", this, null);
    }

    
    public MxNDArray transpose(int... dimensions) {
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
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axes", dimensions);
        return invoke(getParent(), "_np_transpose", this, params);
    }

    
    public MxNDArray broadcast(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.setShape(shape);
        return invoke(getParent(), "_npi_broadcast_to", this, params);
    }

    
    public MxNDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmax", this, null);
    }

    
    public MxNDArray argMax(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmax", this, params);
    }

    
    public MxNDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty MxNDArray");
        }
        return invoke(getParent(), "_npi_argmin", this, null);
    }

    
    public MxNDArray argMin(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return invoke(getParent(), "_npi_argmin", this, params);
    }

    
    public MxNDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    
    public MxNDArray percentile(Number percentile, int[] dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    
    public MxNDArray median() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    
    public MxNDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    
    public MxNDArray nonzero() {
        MxNDArray thisArr =
                (getDataType() == DataType.BOOLEAN) ? toType(DataType.INT32, false) : this;
        return invoke(getParent(), "_npx_nonzero", thisArr, null);
    }

    
    public MxNDArray erfinv() {
        return invoke(getParent(), "erfinv", this, null);
    }

    
    public MxNDArray norm(boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.add("flag", -2);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_norm", this, params);
    }

    
    public MxNDArray norm(int ord, int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addParam("ord", (double) ord);
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return invoke(getParent(), "_npi_norm", this, params);
    }

//    public MxNDArray oneHot(int depth) {
//        return LazyNDArray.super.oneHot(depth);
//    }

    
    public MxNDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.add("depth", depth);
        params.add("on_value", onValue);
        params.add("off_value", offValue);
        params.add("dtype", dataType);
        return invoke(getParent(), "_npx_one_hot", this, params).toType(dataType, false);
    }

    
    public MxNDArray batchDot(MxNDArray other) {
        return invoke(getParent(), "_npx_batch_dot", new MxNDArray[] {this, other}, null);
    }

    
    public MxNDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    @Override
    public void close() {
        if (!getClosed()) {
            // release sub resources
            super.close();
            // release itself
            if (this.getHandle() != null) {
                JnaUtils.freeNdArray(this.getHandle());
            }
            setClosed();
        }
    }
    
    public boolean isEmpty() {
        return getShape().size() == 0;
    }

    boolean isSparse() {
        return getSparseFormat() != SparseFormat.DENSE;
    }

    MxNDArray booleanMask(MxNDArray index) {
        return booleanMask(index, 0);
    }

    boolean shapeEquals(MxNDArray other) {
        return getShape().equals(other.getShape());
    }

    public static MxNDList invoke(MxResource parent, String operation, MxNDList src, PairList<String, ?> params) {
        return new MxNDList(JnaUtils.op(operation).invoke(parent, src.toArray(EMPTY), params));
    }
    
    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link MxNDList} of source {@link MxNDArray}
     * @param dest the {@link MxNDList} to save output to
     * @param params the parameters to be passed to the native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     */
    public static void invoke(String operation, MxNDList src, MxNDList dest, PairList<String, ?> params) {
        invoke(operation, src.toArray(EMPTY), dest.toArray(EMPTY), params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the array of source {@link MxNDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link MxNDArray}
     */
    public static MxNDArray invoke(MxResource parent, String operation, MxNDArray[] src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(parent, src, params)[0];
    }

    public static void invoke(
            String operation, MxNDArray[] src, MxNDArray[] dest, PairList<String, ?> params) {
        JnaUtils.op(operation).invoke(src, dest, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the source {@link MxNDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link MxNDArray}

     */
    public static MxNDArray invoke(MxResource parent, String operation, MxNDArray src, PairList<String, ?> params) {
        return invoke(parent, operation, new MxNDArray[] {src}, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link MxNDArray}

     */
    public static MxNDArray invoke(MxResource parent, String operation, PairList<String, ?> params) {
        return invoke(parent, operation, EMPTY, params);
    }

    /**
     * Encodes {@code MxNDArray} to byte array.
     *
     * @return byte array
     */
    public byte[] encode() {
        return MxNDSerializer.encode(this);
    }


    public static MxNDArray create(MxResource parent, Number data) {
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
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and float
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, float[] data, Shape shape) {
        return create(parent, FloatBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and int
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, int[] data, Shape shape) {
        return create(parent, IntBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and
     * double array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, double[] data, Shape shape) {
        return create(parent, DoubleBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and long
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, long[] data, Shape shape) {
        return create(parent, LongBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and byte
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, byte[] data, Shape shape) {
        return create(parent, ByteBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and
     * boolean array.
     *
     * @param data the boolean array that needs to be set
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public  static MxNDArray create(MxResource parent, boolean[] data, Shape shape) {
        byte[] byteData = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            byteData[i] = (byte) (data[i] ? 1 : 0);
        }
        return create(parent, ByteBuffer.wrap(byteData), shape, DataType.BOOLEAN);
    }
    
    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the float that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, float data) {
        return create(parent, new float[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the float data that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, int data) {
        return create(parent, new int[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the double data that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, double data) {
        return create(parent, new double[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the long data that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, long data) {
        return create(parent, new long[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the byte data that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, byte data) {
        return create(parent, new byte[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link MxNDArray}.
     *
     * @param data the boolean data that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, boolean data) {
        
        return create(parent, new boolean[] {data}, new Shape());
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, float[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, int[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, double[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, long[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, byte[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link MxNDArray}.
     *
     * @param data the bool array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, boolean[] data) {
        return create(parent, data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 2D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, float[][] data) {
        FloatBuffer buffer = FloatBuffer.allocate(data.length * data[0].length);
        for (float[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, int[][] data) {
        IntBuffer buffer = IntBuffer.allocate(data.length * data[0].length);
        for (int[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, double[][] data) {
        DoubleBuffer buffer = DoubleBuffer.allocate(data.length * data[0].length);
        for (double[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, long[][] data) {
        LongBuffer buffer = LongBuffer.allocate(data.length * data[0].length);
        for (long[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link MxNDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, byte[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * data[0].length);
        for (byte[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(parent, buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link MxNDArray}.
     *
     * @param data the boolean array that needs to be set
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray create(MxResource parent, boolean[][] data) {
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
     * Creates and initializes a {@link MxNDArray} with specified {@link Shape}.
     *
     * <p>{@link DataType} of the MxNDArray will determined by type of Buffer.
     *
     * @param data the data to initialize the {@code MxNDArray}
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    static MxNDArray create(MxResource parent, Buffer data, Shape shape) {
        DataType dataType = DataType.fromBuffer(data);
        return create(parent, data, shape, dataType);
    }

    static MxNDArray create(MxResource parent, Buffer data, Shape shape, DataType dataType) {
        MxNDArray array = create(parent, shape, dataType, Device.defaultIfNull());
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
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @param dataType the {@link DataType} of the {@link MxNDArray}
     * @param device the {@link Device} of the {@link MxNDArray}
     * @return the drawn samples {@link MxNDArray}
     */
    public static MxNDArray randomUniform(
            MxResource parent, float low, float high, Shape shape, DataType dataType, Device device) {
        MxOpParams params = new MxOpParams();
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
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @param dataType the {@link DataType} of the {@link MxNDArray}
     * @return the drawn samples {@link MxNDArray}
     */
    private static MxNDArray randomUniform(MxResource parent, float low, float high, Shape shape, DataType dataType) {
        return randomUniform(parent, low, high, shape, dataType, Device.defaultIfNull(null));
    }

    public static MxNDArray randomNormal(
            MxResource parent, float loc, float scale, Shape shape, DataType dataType, Device device) {
        if (device == null) {
            return randomNormal(parent, loc, scale, shape, dataType);
        }
        return randomNormal(parent, loc, scale, shape, dataType);
    }

    public static MxNDArray randomNormal(MxResource parent, float loc, float scale, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
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
     double[] toDoubleArray() {
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
    float[] toFloatArray() {
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
    int[] toIntArray() {
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
    long[] toLongArray() {
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
    int[] toUint8Array() {
        ByteBuffer bb = toByteBuffer();
        int[] buf = new int[bb.remaining()];
        for (int i = 0; i < buf.length; ++i) {
            buf[i] = bb.get() & 0xff;
        }
        return buf;
    }


}
