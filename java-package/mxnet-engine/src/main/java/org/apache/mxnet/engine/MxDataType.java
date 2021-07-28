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

package org.apache.mxnet.engine;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.ndarray.types.DataType;

/** Helper to convert between {@link DataType} and the MXNet internal DataTypes. */
public final class MxDataType {

    private static Map<DataType, String> toMx = createMapToMx();
    private static Map<String, DataType> fromMx = createMapFromMx();

    private MxDataType() {}

    private static Map<DataType, String> createMapToMx() {
        Map<DataType, String> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, "float32");
        map.put(DataType.FLOAT64, "float64");
        map.put(DataType.INT32, "int32");
        map.put(DataType.INT64, "int64");
        map.put(DataType.UINT8, "uint8");
        return map;
    }

    private static Map<String, DataType> createMapFromMx() {
        Map<String, DataType> map = new ConcurrentHashMap<>();
        map.put("float32", DataType.FLOAT32);
        map.put("float64", DataType.FLOAT64);
        map.put("int32", DataType.INT32);
        map.put("int64", DataType.INT64);
        map.put("uint8", DataType.UINT8);
        return map;
    }

    /**
     * Converts a MXNet type String into a {@link DataType}.
     *
     * @param mxType the type String to convert
     * @return the {@link DataType}
     */
    public static DataType fromMx(String mxType) {
        return fromMx.get(mxType);
    }

    /**
     * Converts a {@link DataType} into the corresponding MXNet type String.
     *
     * @param jType the java {@link DataType} to convert
     * @return the converted MXNet type string
     */
    public static String toMx(DataType jType) {
        return toMx.get(jType);
    }
}
