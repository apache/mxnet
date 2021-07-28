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

package org.apache.mxnet.util;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDSerializer;

public class NDArrayUtils {

    /**
     * Decodes {@link NDArray} through byte array.
     *
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
     * @param is input stream data to load from
     * @return {@link NDArray}
     * @throws IOException data is not readable
     */
    public static NDArray decode(MxResource parent, InputStream is) throws IOException {
        return NDSerializer.decode(parent, is);
    }
}
