package org.apache.mxnet.util;

import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDSerializer;
import org.apache.mxnet.engine.MxResource;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

public class MxNDArrayUtils {

    /**
     * Decodes {@link MxNDArray} through byte array.
     *
     * @param bytes byte array to load from
     * @return {@link MxNDArray}
     */
    static MxNDArray decode(MxResource parent, byte[] bytes) {
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))) {
            return MxNDSerializer.decode(parent, dis);
        } catch (IOException e) {
            throw new IllegalArgumentException("NDArray decoding failed", e);
        }
    }

    /**
     * Decodes {@link MxNDArray} through {@link DataInputStream}.
     *
     * @param is input stream data to load from
     * @return {@link MxNDArray}
     * @throws IOException data is not readable
     */
    public static MxNDArray decode(MxResource parent, InputStream is) throws IOException {
        return MxNDSerializer.decode(parent, is);
    }


}
