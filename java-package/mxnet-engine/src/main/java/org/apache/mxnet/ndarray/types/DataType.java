package org.apache.mxnet.ndarray.types;

import org.apache.mxnet.ndarray.MxNDArray;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/** An enum representing the underlying {@link MxNDArray}'s data type. */
public enum DataType {
    FLOAT32(Format.FLOATING, 4),
    FLOAT64(Format.FLOATING, 8),
    FLOAT16(Format.FLOATING, 2),
    UINT8(Format.UINT, 1),
    INT32(Format.INT, 4),
    INT8(Format.INT, 1),
    INT64(Format.INT, 8),
    BOOLEAN(Format.BOOLEAN, 1),
    UNKNOWN(Format.UNKNOWN, 0),
    STRING(Format.STRING, -1);

    /** The general data type format categories. */
    public enum Format {
        FLOATING,
        UINT,
        INT,
        BOOLEAN,
        STRING,
        UNKNOWN
    }

    private Format format;
    private int numOfBytes;

    DataType(Format format, int numOfBytes) {
        this.format = format;
        this.numOfBytes = numOfBytes;
    }

    /**
     * Returns the number of bytes for each element.
     *
     * @return the number of bytes for each element
     */
    public int getNumOfBytes() {
        return numOfBytes;
    }

    /**
     * Returns the format of the data type.
     *
     * @return the format of the data type
     */
    public Format getFormat() {
        return format;
    }

    /**
     * Checks whether it is a floating data type.
     *
     * @return whether it is a floating data type
     */
    public boolean isFloating() {
        return format == Format.FLOATING;
    }

    /**
     * Checks whether it is an integer data type.
     *
     * @return whether it is an integer type
     */
    public boolean isInteger() {
        return format == Format.UINT || format == Format.INT;
    }

    /**
     * Returns the data type to use for a data buffer.
     *
     * @param data the buffer to analyze
     * @return the data type for the buffer
     */
    public static DataType fromBuffer(Buffer data) {
        if (data instanceof FloatBuffer) {
            return DataType.FLOAT32;
        } else if (data instanceof DoubleBuffer) {
            return DataType.FLOAT64;
        } else if (data instanceof IntBuffer) {
            return DataType.INT32;
        } else if (data instanceof LongBuffer) {
            return DataType.INT64;
        } else if (data instanceof ByteBuffer) {
            return DataType.INT8;
        } else {
            throw new IllegalArgumentException(
                    "Unsupported buffer type: " + data.getClass().getSimpleName());
        }
    }

    /**
     * Converts a {@link ByteBuffer} to a buffer for this data type.
     *
     * @param data the buffer to convert
     * @return the converted buffer
     */
    public Buffer asDataType(ByteBuffer data) {
        switch (this) {
            case FLOAT32:
                return data.asFloatBuffer();
            case FLOAT64:
                return data.asDoubleBuffer();
            case INT32:
                return data.asIntBuffer();
            case INT64:
                return data.asLongBuffer();
            case UINT8:
            case INT8:
            case FLOAT16:
            case UNKNOWN:
            default:
                return data;
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return name().toLowerCase();
    }
}