package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;
import org.apache.mxnet.util.PairList;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

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
        mxNDArrayEx = new MxNDArrayEx(this);
    }

    MxNDArray(MxResource parent, Pointer handle, SparseFormat fmt) {
        this(parent, handle);
        this.sparseFormat = fmt;
        mxNDArrayEx = new MxNDArrayEx(this);
        parent.addSubResource(this);
    }

    public MxNDArray create(Pointer handle) {
        if (version >= 10700) {
            return new MxNDArray(this, handle);
        }
        // TODO
        return null;
    }

    public MxNDArray create(Pointer handle, SparseFormat fmt) {
        return new MxNDArray(this, handle, fmt);
    }

    public MxNDArray create(Shape shape, DataType dataType, Device device, boolean hasGradient) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), hasGradient);
        return new MxNDArray(this, handle, device, shape, dataType, hasGradient);

    }
    public MxNDArray create(Shape shape, DataType dataType, Device device) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        return new MxNDArray(this, handle, device, shape, dataType, false);

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
        MxNDArray array = create(shape, dataType, device);
        array.setName(name);
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
                hasGradient() ? (MxNDArray) getGradient() : createGradient(getSparseFormat());
        // DJL go with write as only MXNet support GradReq
        int gradReqValue = requiresGrad ? GradReq.WRITE.getValue() : GradReq.NULL.getValue();
        IntBuffer gradReqBuffer = IntBuffer.allocate(1);
        gradReqBuffer.put(0, gradReqValue);
        JnaUtils.autogradMarkVariables(1, getHandle(), gradReqBuffer, grad.getHandle());
        hasGradient = requiresGrad;
        grad.close();
    }

    private MxNDArray createGradient(SparseFormat format) {
        try (MxNDArray zeros = this.zeros(getShape(), getDataType(), getDevice())) {
            return (MxNDArray) zeros.toSparse(format);
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
        return getManager().invoke(opName, params);
    }

    
    public MxNDArray getGradient() {
        if (!hasGradient()) {
            throw new IllegalStateException(
                    "No gradient attached to this NDArray, please call array.requiredGradient()"
                            + "on your MxNDArray or block.setInitializer() on your Block");
        }
        Pointer pointer = JnaUtils.getGradient(getHandle());
        return getManager().create(pointer);
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
        return getManager().create(pointer);
    }

    
    public String[] toStringArray() {
        throw new UnsupportedOperationException("String MxNDArray is not supported!");
    }

    
    public ByteBuffer toByteBuffer() {
        if (getSparseFormat() != SparseFormat.DENSE) {
            throw new IllegalStateException("Require Dense NDArray, actual " + getSparseFormat());
        }
        Shape sh = getShape();
        MxDataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        ByteBuffer bb = getManager().allocateDirect(Math.toIntExact(len));
        Pointer pointer = Native.getDirectBufferPointer(bb);
        JnaUtils.syncCopyToCPU(getHandle(), pointer, Math.toIntExact(product));
        return bb;
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
        MxDataType inputType = MxDataType.fromBuffer(data);
        validate(inputType);

        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = getManager().allocateDirect(size * numOfBytes);

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

    private void validate(MxDataType inputType) {
        if (getDataType() != inputType
                && ((dataType != MxDataType.UINT8 && dataType != MxDataType.BOOLEAN)
                || inputType != MxDataType.INT8)) {
            // Infer DataType from Buffer always return INT8, make this two special case that
            // allows set UINT8 and BOOL array with regular ByteBuffer.
            throw new IllegalStateException(
                    "DataType mismatch, required: " + dataType + ", actual: " + inputType);
        }
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
        try (MxNDArray reshaped = this.reshape(reshape);
             MxNDArray reshapedIndex = index.toType(MxDataType.INT32, false).reshape(-1);
             MxNDArray result =
                     getManager().invoke(
                             "_npi_boolean_mask",
                             new NDArray[] {reshaped, reshapedIndex},
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
        return getManager().invoke("_npx_sequence_mask", new NDList(this, sequenceLength), params)
                .head();
    }

    
    public MxNDArray sequenceMask(NDArray sequenceLength) {
        return sequenceMask(sequenceLength, 0);
    }

    
    public MxNDArray zerosLike() {
        MxOpParams params = new MxOpParams();
        params.addParam("fill_value", 0);
        return getManager().invoke("_npi_full_like", this, params);
    }

    
    public MxNDArray onesLike() {
        MxOpParams params = new MxOpParams();
        params.addParam("fill_value", 1);
        return getManager().invoke("_npi_full_like", this, params);
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
        try (NDArray result = eq(other).toType(MxDataType.INT32, false)) {
            return result.all().getBoolean();
        }
    }

    
    public MxNDArray eq(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_equal_scalar", this, params);
    }

    
    public MxNDArray eq(NDArray other) {
        return getManager().invoke("_npi_equal", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray neq(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_not_equal_scalar", this, params);
    }

    
    public MxNDArray neq(NDArray other) {
        return getManager().invoke("_npi_not_equal", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray gt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return getManager().invoke("_npi_greater_scalar", this, params);
    }

    
    public MxNDArray gt(NDArray other) {
        return getManager().invoke("_npi_greater", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray gte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return getManager().invoke("_npi_greater_equal_scalar", this, params);
    }

    
    public MxNDArray gte(NDArray other) {
        return getManager().invoke("_npi_greater_equal", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray lt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return getManager().invoke("_npi_less_scalar", this, params);
    }

    
    public MxNDArray lt(NDArray other) {
        return getManager().invoke("_npi_less", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray lte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return getManager().invoke("_npi_less_equal_scalar", this, params);
    }

    
    public MxNDArray lte(NDArray other) {
        return getManager().invoke("_npi_less_equal", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray add(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_add_scalar", this, params);
    }

    
    public MxNDArray add(NDArray other) {
        return getManager().invoke("_npi_add", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray sub(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_subtract_scalar", this, params);
    }

    
    public MxNDArray sub(NDArray other) {
        return getManager().invoke("_npi_subtract", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray mul(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_multiply_scalar", this, params);
    }

    
    public MxNDArray mul(NDArray other) {
        return getManager().invoke("_npi_multiply", new NDArray[] {this, other}, null);
    }


    
    public MxNDArray div(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_true_divide_scalar", this, params);
    }

    
    public MxNDArray div(NDArray other) {
        return getManager().invoke("_npi_true_divide", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray mod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_mod_scalar", this, params);
    }

    
    public MxNDArray mod(NDArray other) {
        return getManager().invoke("_npi_mod", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray pow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_power_scalar", this, params);
    }

    
    public MxNDArray pow(NDArray other) {
        return getManager().invoke("_npi_power", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray addi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_npi_add_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray addi(NDArray other) {
        getManager().invoke("_npi_add", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray subi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_npi_subtract_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray subi(NDArray other) {
        getManager().invoke("_npi_subtract", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray muli(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_npi_multiply_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray muli(NDArray other) {
        getManager().invoke("_npi_multiply", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray divi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke(
                "_npi_true_divide_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray divi(NDArray other) {
        getManager().invoke("_npi_true_divide", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray modi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_npi_mod_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray modi(NDArray other) {
        getManager().invoke("_npi_mod", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray powi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        getManager().invoke("_npi_power_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    
    public MxNDArray powi(NDArray other) {
        getManager().invoke("_npi_power", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray sign() {
        return getManager().invoke("_npi_sign", this, null);
    }

    
    public MxNDArray signi() {
        getManager().invoke("_npi_sign", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray maximum(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_maximum_scalar", this, params);
    }

    
    public MxNDArray maximum(NDArray other) {
        return getManager().invoke("_npi_maximum", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray minimum(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return getManager().invoke("_npi_minimum_scalar", this, params);
    }

    
    public MxNDArray minimum(NDArray other) {
        return getManager().invoke("_npi_minimum", new NDArray[] {this, other}, null);
    }


    @Override
    public MxNDArray neg() {
        return getManager().invoke("_npi_negative", this, null);
    }

    
    public MxNDArray negi() {
        getManager().invoke("_npi_negative", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    
    public MxNDArray abs() {
        return getManager().invoke("_npi_absolute", this, null);
    }

    
    public MxNDArray square() {
        return getManager().invoke("_npi_square", this, null);
    }

    
    public MxNDArray sqrt() {
        return getManager().invoke("_npi_sqrt", this, null);
    }

    
    public MxNDArray cbrt() {
        return getManager().invoke("_npi_cbrt", this, null);
    }

    
    public MxNDArray floor() {
        return getManager().invoke("_npi_floor", this, null);
    }

    
    public MxNDArray ceil() {
        return getManager().invoke("_npi_ceil", this, null);
    }

    
    public MxNDArray round() {
        return getManager().invoke("round", this, null);
    }

    
    public MxNDArray trunc() {
        return getManager().invoke("_npi_trunc", this, null);
    }

    
    public MxNDArray exp() {
        return getManager().invoke("_npi_exp", this, null);
    }

    
    public MxNDArray log() {
        return getManager().invoke("_npi_log", this, null);
    }

    
    public MxNDArray log10() {
        return getManager().invoke("_npi_log10", this, null);
    }

    
    public MxNDArray log2() {
        return getManager().invoke("_npi_log2", this, null);
    }

    
    public MxNDArray sin() {
        return getManager().invoke("_npi_sin", this, null);
    }

    
    public MxNDArray cos() {
        return getManager().invoke("_npi_cos", this, null);
    }

    
    public MxNDArray tan() {
        return getManager().invoke("_npi_tan", this, null);
    }

    
    public MxNDArray asin() {
        return getManager().invoke("_npi_arcsin", this, null);
    }

    
    public MxNDArray acos() {
        return getManager().invoke("_npi_arccos", this, null);
    }

    
    public MxNDArray atan() {
        return getManager().invoke("_npi_arctan", this, null);
    }

    
    public MxNDArray sinh() {
        return getManager().invoke("_npi_sinh", this, null);
    }

    
    public MxNDArray cosh() {
        return getManager().invoke("_npi_cosh", this, null);
    }

    
    public MxNDArray tanh() {
        return getManager().invoke("_npi_tanh", this, null);
    }

    
    public MxNDArray asinh() {
        return getManager().invoke("_npi_arcsinh", this, null);
    }

    
    public MxNDArray acosh() {
        return getManager().invoke("_npi_arccosh", this, null);
    }

    
    public MxNDArray atanh() {
        return getManager().invoke("_npi_arctanh", this, null);
    }

    
    public MxNDArray toDegrees() {
        return getManager().invoke("_npi_degrees", this, null);
    }

    
    public MxNDArray toRadians() {
        return getManager().invoke("_npi_radians", this, null);
    }


    
    public MxNDArray max() {
        return getManager().invoke("_np_max", this, null);
    }

    
    public MxNDArray max(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return getManager().invoke("_np_max", this, params);
    }

    
    public MxNDArray max(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_np_max", this, params);
    }

    
    public MxNDArray min() {
        return getManager().invoke("_np_min", this, null);
    }

    
    public MxNDArray min(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_np_min", this, params);
    }

    
    public MxNDArray sum() {
        // TODO current windows doesn't support boolean NDArray
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            MxDataType target = getDataType();
            if (!target.isFloating()) {
                try (NDArray thisArr = toType(MxDataType.FLOAT32, false)) {
                    if (target == MxDataType.BOOLEAN) {
                        target = MxDataType.INT64;
                    }
                    try (NDArray array = getManager().invoke("_np_sum", thisArr, null)) {
                        return array.toType(target, false);
                    }
                }
            }
        }
        return getManager().invoke("_np_sum", this, null);
    }

    
    public MxNDArray sum(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_np_sum", this, params);
    }

    
    public MxNDArray prod() {
        return getManager().invoke("_np_prod", this, null);
    }

    
    public MxNDArray prod(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_np_prod", this, params);
    }

    
    public MxNDArray mean() {
        return getManager().invoke("_npi_mean", this, null);
    }

    
    public MxNDArray mean(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_npi_mean", this, params);
    }

    
    public MxNDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axes", axes);
        params.addParam("k", times);
        return getManager().invoke("_npi_rot90", this, params);
    }

    
    public MxNDArray trace(int offset, int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("offset", offset);
        params.addParam("axis1", axis1);
        params.addParam("axis2", axis2);
        return getManager().invoke("_np_trace", this, params);
    }

    
    public NDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(this);
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
        return getManager().invoke("_npi_split", new NDList(this), params);
    }

    
    public MxNDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    
    public MxNDArray reshape(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.addParam("newshape", shape);
        return getManager().invoke("_np_reshape", this, params);
    }

    
    public MxNDArray expandDims(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npi_expand_dims", this, params);
    }

    
    public MxNDArray squeeze() {
        return getManager().invoke("_np_squeeze", this, null);
    }

    
    public MxNDArray squeeze(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return getManager().invoke("_np_squeeze", this, params);
    }

    
    public MxNDArray logicalAnd(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == MxDataType.BOOLEAN) ? toType(MxDataType.INT32, false) : this;
        other =
                (other.getDataType() == MxDataType.BOOLEAN)
                        ? other.toType(MxDataType.INT32, false)
                        : other;
        return getManager().invoke("broadcast_logical_and", new NDArray[] {thisArr, other}, null)
                .toType(MxDataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalOr(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == MxDataType.BOOLEAN) ? toType(MxDataType.INT32, false) : this;
        other =
                (other.getDataType() == MxDataType.BOOLEAN)
                        ? other.toType(MxDataType.INT32, false)
                        : other;
        return getManager().invoke("broadcast_logical_or", new NDArray[] {thisArr, other}, null)
                .toType(MxDataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalXor(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        MxNDArray thisArr =
                (getDataType() == MxDataType.BOOLEAN) ? toType(MxDataType.INT32, false) : this;
        other =
                (other.getDataType() == MxDataType.BOOLEAN)
                        ? other.toType(MxDataType.INT32, false)
                        : other;
        return getManager().invoke("broadcast_logical_xor", new NDArray[] {thisArr, other}, null)
                .toType(MxDataType.BOOLEAN, false);
    }

    
    public MxNDArray logicalNot() {
        return getManager().invoke("_npi_logical_not", this, null);
    }

    
    public MxNDArray argSort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(MxDataType.INT64);
        return getManager().invoke("_npi_argsort", this, params);
    }

    
    public MxNDArray sort(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npi_sort", this, params);
    }

    
    public MxNDArray sort() {
        return getManager().invoke("_npi_sort", this, null);
    }

    
    public MxNDArray softmax(int axis) {
        // MXNet softmax op bug on GPU
        if (isEmpty()) {
            return getManager().create(getShape(), MxDataType.FLOAT32, getDevice());
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npx_softmax", this, params);
    }

    
    public MxNDArray logSoftmax(int axis) {
        // MXNet logsoftmax op bug on GPU
        if (isEmpty()) {
            return getManager().create(getShape(), MxDataType.FLOAT32, getDevice());
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npx_log_softmax", this, params);
    }

    
    public MxNDArray cumSum() {
        return getManager().invoke("_np_cumsum", this, null);
    }

    
    public MxNDArray cumSum(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_np_cumsum", this, params);
    }

    
    public void intern(NDArray replaced) {
        MxNDArray arr = (MxNDArray) replaced;
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
        return getManager().invoke("_npi_isnan", this, null);
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
        return getManager().invoke("cast_storage", this, params);
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
        return getManager().invoke("_npi_tile", this, params);
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
                array = getManager().invoke("_np_repeat", array, params);
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

    
    public MxNDArray dot(NDArray other) {
        return getManager().invoke("_np_dot", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        return getManager().invoke("_npi_matmul", new NDArray[] {this, other}, null);
    }

    
    public MxNDArray clip(Number min, Number max) {
        MxOpParams params = new MxOpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return getManager().invoke("_npi_clip", this, params);
    }

    
    public MxNDArray swapAxes(int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("dim1", axis1);
        params.addParam("dim2", axis2);
        return getManager().invoke("_npi_swapaxes", this, params);
    }

    
    public MxNDArray flip(int... axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return getManager().invoke("_npi_flip", this, params);
    }

    
    public MxNDArray transpose() {
        return getManager().invoke("_np_transpose", this, null);
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
        return getManager().invoke("_np_transpose", this, params);
    }

    
    public MxNDArray broadcast(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.setShape(shape);
        return getManager().invoke("_npi_broadcast_to", this, params);
    }

    
    public MxNDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        return getManager().invoke("_npi_argmax", this, null);
    }

    
    public MxNDArray argMax(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npi_argmax", this, params);
    }

    
    public MxNDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        return getManager().invoke("_npi_argmin", this, null);
    }

    
    public MxNDArray argMin(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return getManager().invoke("_npi_argmin", this, params);
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
                (getDataType() == MxDataType.BOOLEAN) ? toType(MxDataType.INT32, false) : this;
        return getManager().invoke("_npx_nonzero", thisArr, null);
    }

    
    public MxNDArray erfinv() {
        return getManager().invoke("erfinv", this, null);
    }

    
    public MxNDArray norm(boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.add("flag", -2);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_npi_norm", this, params);
    }

    
    public MxNDArray norm(int ord, int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addParam("ord", (double) ord);
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return getManager().invoke("_npi_norm", this, params);
    }

    @Override
    public MxNDArray oneHot(int depth) {
        return LazyNDArray.super.oneHot(depth);
    }

    
    public MxNDArray oneHot(int depth, float onValue, float offValue, MxDataType dataType) {
        MxOpParams params = new MxOpParams();
        params.add("depth", depth);
        params.add("on_value", onValue);
        params.add("off_value", offValue);
        params.add("dtype", dataType);
        return getManager().invoke("_npx_one_hot", this, params).toType(dataType, false);
    }

    
    public MxNDArray batchDot(MxNDArray other) {
        return getManager().invoke("_npx_batch_dot", new NDArray[] {this, other}, null);
    }

    
    public MxNDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    
    public void close() {

    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link MxNDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public void invoke(String operation, MxNDList src, NDList dest, PairList<String, ?> params) {
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
     * @param src the array of source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray[] src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(this, src, params)[0];
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray src, PairList<String, ?> params) {
        return invoke(operation, new NDArray[] {src}, params);
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
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, PairList<String, ?> params) {
        return invoke(operation, EMPTY, params);
    }

    /**
     * Encodes {@code NDArray} to byte array.
     *
     * @return byte array
     */
    public byte[] encode() {
        return MxNDSerializer.encode(this);
    }


}
