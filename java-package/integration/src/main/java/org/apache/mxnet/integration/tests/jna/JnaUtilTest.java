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

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.SymbolBlock;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.repository.Item;
import org.apache.mxnet.repository.Repository;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void doForwardTest() throws IOException {
        // TODO: replace the Path of model with soft decoding
        try (
                MxResource base = BaseMxResource.getSystemMxResource()
                ) {
            Path modelPath = Repository.initRepository(Item.MLP);
            Path symbolPath = modelPath.resolve("mlp-symbol.json");
            Path paramsPath = modelPath.resolve("mlp-0000.params");
            Symbol symbol  = Symbol.loadSymbol(base, symbolPath);
            SymbolBlock block = new SymbolBlock(base, symbol);
            Device device = Device.defaultIfNull();
            NDList mxNDArray = JnaUtils.loadNdArray(
                    base,
                    paramsPath,
                    Device.defaultIfNull(null));

            // load parameters
            List<Parameter> parameters = block.getAllParameters();
            Map<String, Parameter> map = new LinkedHashMap<>();
            parameters.forEach(p -> map.put(p.getName(), p));

            for (NDArray nd : mxNDArray) {
                String key = nd.getName();
                if (key == null) {
                    throw new IllegalArgumentException("Array names must be present in parameter file");
                }

                String paramName = key.split(":", 2)[1];
                Parameter parameter = map.remove(paramName);
                parameter.setArray(nd);
            }
            block.setInputNames(new ArrayList<>(map.keySet()));

            NDArray arr = NDArray.create(base, new Shape(1, 28, 28), device).ones();
            block.forward(new ParameterStore(base, false, device), new NDList(arr), false, new PairList<>(), device);
            logger.info("Number of MxResource managed by baseMxResource: {}",
                    BaseMxResource.getSystemMxResource().getSubResource().size());
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
            throw e;
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            throw e;
        }
        Assert.assertEquals(BaseMxResource.getSystemMxResource().getSubResource().size(), 0);

    }

    @Test
    public void createNdArray() {
        try {
            try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                int[] originIntegerArray = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                float[] originFlaotArray = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                double[] originDoubleArray = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                long[] originLongArray = new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                boolean[] originBooleanArray = new boolean[]{true, false, false, true, true, true, true, false, false, true, true, true};
                byte[] originByteArray = new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                NDArray intArray = NDArray.create(base, originIntegerArray, new Shape(3, 4));
                NDArray floatArray = NDArray.create(base, originFlaotArray, new Shape(3, 4));
                NDArray doubleArray = NDArray.create(base, originDoubleArray, new Shape(3, 4));
                NDArray longArray = NDArray.create(base, originLongArray, new Shape(3, 4));
                NDArray booleanArray = NDArray.create(base, originBooleanArray, new Shape(3, 4));
                NDArray byteArray = NDArray.create(base, originByteArray, new Shape(3, 4));
                NDArray intArray2 = NDArray.create(base, originIntegerArray);
                NDArray floatArray2 = NDArray.create(base, originFlaotArray);
                NDArray doubleArray2 = NDArray.create(base, originDoubleArray);
                NDArray longArray2 = NDArray.create(base, originLongArray);
                NDArray booleanArray2 = NDArray.create(base, originBooleanArray);
                NDArray byteArray2 = NDArray.create(base, originByteArray);

                Integer[] ndArrayInt = (Integer[]) intArray.toArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt);
                // Float -> Double
                double[] floats = Arrays.stream(floatArray.toArray()).mapToDouble(Number::floatValue).toArray();
                Assert.assertEquals(originDoubleArray, floats);
                Double[] ndArrayDouble = (Double[]) doubleArray.toArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble);
                Long[] ndArrayLong = (Long[]) longArray.toArray();
                Assert.assertEquals(originLongArray, ndArrayLong);
                boolean[] ndArrayBoolean = booleanArray.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean);
                byte[] ndArrayByte = byteArray.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte);


                Integer[] ndArrayInt2 = (Integer[]) intArray2.toArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt2);

                // Float -> Double
                double[] floats2 = Arrays.stream(floatArray2.toArray()).mapToDouble(Number::floatValue).toArray();
                Assert.assertEquals(originDoubleArray, floats2);
                Double[] ndArrayDouble2 = (Double[]) doubleArray2.toArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble2);
                Long[] ndArrayLong2 = (Long[]) longArray2.toArray();
                Assert.assertEquals(originLongArray, ndArrayLong2);
                boolean[] ndArrayBoolean2 = booleanArray2.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean2);
                byte[] ndArrayByte2 = byteArray2.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte2);
            } catch (Exception e) {
                logger.error(e.getMessage());
                e.printStackTrace();
                throw e;
            }
            BaseMxResource base = BaseMxResource.getSystemMxResource();
//            assert base.getSubResource().size() == 0;
        } catch (Exception e) {
            logger.error(e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void loadNdArray() {

        try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                NDList mxNDArray = JnaUtils.loadNdArray(
                        base,
                        Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params"),
                        Device.defaultIfNull(null));

            System.out.println(base.getSubResource().size());
        }
        System.out.println(BaseMxResource.getSystemMxResource().getSubResource().size());

    }
}
