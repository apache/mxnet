package org.apache.mxnet.util;

import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDSerializer;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.MxResourceFactory;

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
    static MxNDArray decode(InputStream is) throws IOException {
        return NDSerializer.decode(this, is);
    }


}
