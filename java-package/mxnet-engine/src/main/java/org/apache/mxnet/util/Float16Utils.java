package org.apache.mxnet.util;

import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDSerializer;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

/** {@code Float16Utils} is a set of utilities for working with float16. */
@SuppressWarnings("PMD.AvoidUsingShortType")
public final class Float16Utils {

    private Float16Utils() {}

    /**
     * Converts a byte buffer of float16 values into a float32 array.
     *
     * @param buffer the buffer of float16 values as bytes.
     * @return an array of float32 values.
     */
    public static float[] fromByteBuffer(ByteBuffer buffer) {
        return fromShortBuffer(buffer.asShortBuffer());
    }

    /**
     * Converts a short buffer of float16 values into a float32 array.
     *
     * @param buffer the buffer of float16 values as shorts.
     * @return an array of float32 values.
     */
    public static float[] fromShortBuffer(ShortBuffer buffer) {
        int index = 0;
        float[] ret = new float[buffer.remaining()];
        while (buffer.hasRemaining()) {
            short value = buffer.get();
            ret[index++] = halfToFloat(value);
        }
        return ret;
    }

    /**
     * Converts an array of float32 values into a byte buffer of float16 values.
     *
     * @param floats an array of float32 values.
     * @return a byte buffer with float16 values represented as shorts (2 bytes each).
     */
    public static ByteBuffer toByteBuffer(float[] floats) {
        ByteBuffer buffer = MxNDSerializer.allocateDirect(floats.length * 2);
        for (float f : floats) {
            short value = floatToHalf(f);
            buffer.putShort(value);
        }
        buffer.rewind();
        return buffer;
    }

    /**
     * Converts a float32 value into a float16 value.
     *
     * @param fVal a float32 value.
     * @return a float16 value represented as a short.
     */
    public static short floatToHalf(float fVal) {
        int bits = Float.floatToIntBits(fVal);
        int sign = bits >>> 16 & 0x8000;
        int val = (bits & 0x7fffffff) + 0x1000;
        if (val >= 0x47800000) {
            if ((bits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) {
                    return (short) (sign | 0x7c00);
                }
                return (short) (sign | 0x7c00 | (bits & 0x007fffff) >>> 13);
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) {
            return (short) (sign | val - 0x38000000 >>> 13);
        }
        if (val < 0x33000000) {
            return (short) sign;
        }
        val = (bits & 0x7fffffff) >>> 23;
        return (short)
                (sign | ((bits & 0x7fffff | 0x800000) + (0x800000 >>> val - 102) >>> 126 - val));
    }

    /**
     * Converts a float16 value into a float32 value.
     *
     * @param half a float16 value represented as a short.
     * @return a float32 value.
     */
    public static float halfToFloat(short half) {
        int mant = half & 0x03ff;
        int exp = half & 0x7c00;
        if (exp == 0x7c00) {
            exp = 0x3fc00;
        } else if (exp != 0) {
            exp += 0x1c000;
            if (mant == 0 && exp > 0x1c400) {
                return Float.intBitsToFloat((half & 0x8000) << 16 | exp << 13);
            }
        } else if (mant != 0) {
            exp = 0x1c400;
            do {
                mant <<= 1;
                exp -= 0x400;
            } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
        }
        return Float.intBitsToFloat((half & 0x8000) << 16 | (exp | mant) << 13);
    }
}

