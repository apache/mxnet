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

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;

/** A interface contains encoding and decoding logic for NDArray. */
public final class NDSerializer {

    static final int BUFFER_SIZE = 81920;
    static final String MAGIC_NUMBER = "NDAR";
    static final int VERSION = 2;

    private NDSerializer() {}

    /**
     * Allocates a new engine specific direct byte buffer.
     *
     * @param capacity the new buffer's capacity, in bytes
     * @return the new byte buffer
     */
    public static ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /**
     * Encodes {@link NDArray} to byte array.
     *
     * @param array the input {@link NDArray}
     * @return byte array
     */
    static byte[] encode(NDArray array) {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream dos = new DataOutputStream(baos);
            // magic string for version identification
            dos.writeUTF(MAGIC_NUMBER);
            dos.writeInt(VERSION);
            String name = array.getName();
            if (name == null) {
                dos.write(0);
            } else {
                dos.write(1);
                dos.writeUTF(name);
            }
            dos.writeUTF(array.getSparseFormat().name());
            dos.writeUTF(array.getDataType().name());

            Shape shape = array.getShape();
            dos.write(shape.getEncoded());

            ByteBuffer bb = array.toByteBuffer();
            int length = bb.remaining();
            dos.writeInt(length);

            if (length > 0) {
                if (length > BUFFER_SIZE) {
                    byte[] buf = new byte[BUFFER_SIZE];
                    while (length > BUFFER_SIZE) {
                        bb.get(buf);
                        dos.write(buf);
                        length = bb.remaining();
                    }
                }

                byte[] buf = new byte[length];
                bb.get(buf);
                dos.write(buf);
            }
            dos.flush();
            return baos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("This should never happen", e);
        }
    }

    /**
     * Decodes {@link NDArray} through {@link DataInputStream}.
     *
     * @param parent the parent MxResource object which create the returned object
     * @param is input stream data to load from
     * @return {@link NDArray}
     * @throws IOException data is not readable
     */
    public static NDArray decode(MxResource parent, InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        if (!"NDAR".equals(dis.readUTF())) {
            throw new IllegalArgumentException("Malformed NDArray data");
        }

        // NDArray encode version
        int version = dis.readInt();
        if (version < 1 || version > VERSION) {
            throw new IllegalArgumentException("Unexpected NDArray encode version " + version);
        }

        String name = null;
        if (version > 1) {
            byte flag = dis.readByte();
            if (flag == 1) {
                name = dis.readUTF();
            }
        }

        dis.readUTF(); // ignore SparseFormat

        // DataType - 1 byte
        DataType dataType = DataType.valueOf(dis.readUTF());

        // Shape
        Shape shape = Shape.decode(dis);

        // Data
        int length = dis.readInt();
        ByteBuffer data = allocateDirect(length);

        if (length > 0) {
            byte[] buf = new byte[BUFFER_SIZE];
            while (length > BUFFER_SIZE) {
                dis.readFully(buf);
                data.put(buf);
                length -= BUFFER_SIZE;
            }

            dis.readFully(buf, 0, length);
            data.put(buf, 0, length);
            data.rewind();
        }
        NDArray array = NDArray.create(parent, dataType.asDataType(data), shape, dataType);
        array.setName(name);
        return array;
    }
}
