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

package org.apache.mxnet.integration.tests.jna;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.nn.SymbolBlock;
import org.apache.mxnet.repository.Item;
import org.apache.mxnet.repository.Repository;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void doForwardTest() throws IOException {
        try (MxResource base = BaseMxResource.getSystemMxResource()) {
            Path modelPath = Repository.initRepository(Item.MLP);
            Path symbolPath = modelPath.resolve("mlp-symbol.json");
            Path paramsPath = modelPath.resolve("mlp-0000.params");
            Symbol symbol = Symbol.loadSymbol(base, symbolPath);
            SymbolBlock block = new SymbolBlock(base, symbol);
            Device device = Device.defaultIfNull();
            NDList mxNDArray = JnaUtils.loadNdArray(base, paramsPath, Device.defaultIfNull(null));

            // load parameters
            List<Parameter> parameters = block.getAllParameters();
            Map<String, Parameter> map = new ConcurrentHashMap<>();
            parameters.forEach(p -> map.put(p.getName(), p));

            for (NDArray nd : mxNDArray) {
                String key = nd.getName();
                if (key == null) {
                    throw new IllegalArgumentException(
                            "Array names must be present in parameter file");
                }

                String paramName = key.split(":", 2)[1];
                Parameter parameter = map.remove(paramName);
                parameter.setArray(nd);
            }
            block.setInputNames(new ArrayList<>(map.keySet()));

            NDArray arr = NDArray.create(base, new Shape(1, 28, 28), device).ones();
            block.forward(new NDList(arr), new PairList<>(), device);
            logger.info(
                    "Number of MxResource managed by baseMxResource: {}",
                    BaseMxResource.getSystemMxResource().getSubResource().size());
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
            throw e;
        }
        Assert.assertEquals(BaseMxResource.getSystemMxResource().getSubResource().size(), 0);
    }

    @Test
    public void createNdArray() {
        try {
            try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                int[] originIntegerArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                float[] originFloatArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                double[] originDoubleArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                long[] originLongArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                boolean[] originBooleanArray = {
                    true, false, false, true, true, true, true, false, false, true, true, true
                };
                byte[] originByteArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                NDArray intArray = NDArray.create(base, originIntegerArray, new Shape(3, 4));
                NDArray floatArray = NDArray.create(base, originFloatArray, new Shape(3, 4));
                NDArray doubleArray = NDArray.create(base, originDoubleArray, new Shape(3, 4));
                NDArray longArray = NDArray.create(base, originLongArray, new Shape(3, 4));
                NDArray booleanArray = NDArray.create(base, originBooleanArray, new Shape(3, 4));
                NDArray byteArray = NDArray.create(base, originByteArray, new Shape(3, 4));
                NDArray intArray2 = NDArray.create(base, originIntegerArray);
                NDArray floatArray2 = NDArray.create(base, originFloatArray);
                NDArray doubleArray2 = NDArray.create(base, originDoubleArray);
                NDArray longArray2 = NDArray.create(base, originLongArray);
                NDArray booleanArray2 = NDArray.create(base, originBooleanArray);
                NDArray byteArray2 = NDArray.create(base, originByteArray);

                int[] ndArrayInt = intArray.toIntArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt);
                // Float -> Double
                float[] floats = floatArray.toFloatArray();
                Assert.assertEquals(originFloatArray, floats);
                double[] ndArrayDouble = doubleArray.toDoubleArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble);
                long[] ndArrayLong = longArray.toLongArray();
                Assert.assertEquals(originLongArray, ndArrayLong);
                boolean[] ndArrayBoolean = booleanArray.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean);
                byte[] ndArrayByte = byteArray.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte);

                int[] ndArrayInt2 = intArray2.toIntArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt2);

                // Float -> Double
                float[] floats2 = floatArray2.toFloatArray();
                Assert.assertEquals(originFloatArray, floats2);
                double[] ndArrayDouble2 = doubleArray2.toDoubleArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble2);
                long[] ndArrayLong2 = longArray2.toLongArray();
                Assert.assertEquals(originLongArray, ndArrayLong2);
                boolean[] ndArrayBoolean2 = booleanArray2.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean2);
                byte[] ndArrayByte2 = byteArray2.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte2);
            } catch (ClassCastException e) {
                logger.error(e.getMessage());
                throw e;
            }
            BaseMxResource base = BaseMxResource.getSystemMxResource();
            Assert.assertEquals(base.getSubResource().size(), 0);
        } catch (ClassCastException e) {
            logger.error(e.getMessage());
            throw e;
        }
    }

    //    @Test
    //    public void loadNdArray() {
    //
    //        try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
    //            NDList mxNDArray =
    //                    JnaUtils.loadNdArray(
    //                            base,
    //                            Paths.get(
    //
    // "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params"),
    //                            Device.defaultIfNull(null));
    //            logger.info(mxNDArray.toString());
    //            logger.info(
    //                    String.format(
    //                            "The amount of sub resources managed by BaseMxResource: %s",
    //                            base.getSubResource().size()));
    //        }
    //        logger.info(
    //                String.format(
    //                        "The amount of sub resources managed by BaseMxResource: %s",
    //                        BaseMxResource.getSystemMxResource().getSubResource().size()));
    //    }
}
