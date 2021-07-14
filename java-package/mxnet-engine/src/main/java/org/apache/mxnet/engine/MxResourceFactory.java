package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.ndarray.types.SparseFormat;

import java.nio.Buffer;

public class MxResourceFactory {

    public static MxNDArray createNDArray(MxResource parent, Pointer pointer) {
        return new MxNDArray(parent, pointer);
    }

    public static MxNDArray createNDArray(MxResource parent, DataType dataType, Shape shape, Device device) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        return new MxNDArray(parent, handle, device, shape, dataType, false);
    }

    // create the MxNDArray with the baseMxResource as it's parent
    public static MxNDArray createNDArray(Pointer handle) {
        return new MxNDArray(handle);
    }

    /**
     * Creates and initializes a {@link MxNDArray} with specified {@link Shape}.
     *
     * <p>{@link DataType} of the NDArray will determined by type of Buffer.
     *
     * @param data the data to initialize the {@code MxNDArray}
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray createNDArray(Buffer data, Shape shape) {
        DataType dataType = DataType.fromBuffer(data);
        return createNDArray(data, shape, dataType);
    }

    /**
     * Creates and initializes an instance of {@link MxNDArray} with specified {@link Shape} and
     * {@link DataType}.
     *
     * @param data the data to initialize the {@link MxNDArray}
     * @param shape the {@link Shape} of the {@link MxNDArray}
     * @param dataType the {@link DataType} of the {@link MxNDArray}
     * @return a new instance of {@link MxNDArray}
     */
    public static MxNDArray createNDArray(Buffer data, Shape shape, DataType dataType) {
        MxNDArray array = createNDArray(data, shape);
        array.set(data);
        return array;
    }

    public static MxNDArray createNDArray(Buffer data, Shape shape, SparseFormat sparseFormat) {
        MxNDArray array = createNDArray(data, shape);
        array.set(data);
        return array;
    }




}
