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

import com.sun.jna.Pointer;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.nn.SymbolBlock;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code CachedOp} is an internal helper that provides the core functionality to execute a
 * {@link SymbolBlock}.
 *
 * <p>We don't recommend users interact with this class directly. Users should use {@link Predictor}
 * instead. CachedOp is an operator that simplifies calling and analyzing the input shape. It
 * requires minimum input to do inference because most of the information can be obtained from the
 * model itself.
 */
public class CachedOp extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private List<Parameter> parameters;
    private PairList<String, Integer> dataIndices;
    private Map<String, Integer> dataIndicesMap;
    private List<Integer> paramIndices;

    /**
     * Creates an instance of {@link CachedOp}.
     *
     * <p>It can be created by using {@link JnaUtils#createCachedOp(SymbolBlock, MxResource)}
     *
     * @param parent the MxResource object to manage this instance of CachedOp
     * @param handle the C handle of the CachedOp
     * @param parameters the parameter values
     * @param paramIndices the parameters required by the model and their corresponding location
     * @param dataIndices the input data names required by the model and their corresponding
     *     location
     */
    public CachedOp(
            MxResource parent,
            Pointer handle,
            List<Parameter> parameters,
            List<Integer> paramIndices,
            PairList<String, Integer> dataIndices) {
        super(parent, handle);
        this.parameters = parameters;
        this.dataIndices = dataIndices;
        this.paramIndices = paramIndices;
        this.dataIndicesMap = dataIndices.toMap();
    }

    /**
     * Assigns inputs to the empty locations of the input NDArray.
     *
     * @param data the input in {@link NDList} format
     * @return an {@link NDList}
     */
    public NDList forward(NDList data) {
        // reset the input data index at the beginning
        NDArray[] allInputsNDArray = new NDArray[parameters.size()];
        // check device of input
        Device device = data.isEmpty() ? Device.defaultIfNull() : data.head().getDevice();
        // fill allInputsNDArray with parameter values on correct device
        for (int index : paramIndices) {
            Parameter parameter = parameters.get(index);
            NDArray value = parameter.getArray();
            if (value == null) {
                throw new NullPointerException("Failed to find parameter from parameterStore");
            }
            value.setDevice(device);
            allInputsNDArray[index] = value;
        }

        // fill allInputsNDArray with data values
        int index = 0;
        for (NDArray array : data) {
            // TODO: NDArray name doesn't match. To confirm the format of input name
            //            String inputName = array.getName().split(":")[1];
            String inputName = array.getName();
            // if inputName not provided, value will follow the default order
            int idx = indexOf(inputName, index++);
            allInputsNDArray[idx] = array;
        }

        // check the input, set as Shape(batchSize) by default
        for (Pair<String, Integer> pair : dataIndices) {
            if (allInputsNDArray[pair.getValue()] == null) {
                // TODO: Do we need to set default to the input?
                long batchSize = data.head().getShape().get(0);
                String key = pair.getKey();
                if (!"prob_label".equals(key) && !"softmax_label".equals(key)) {
                    logger.warn(
                            "Input "
                                    + key
                                    + " not found, set NDArray to Shape("
                                    + batchSize
                                    + ") by default");
                }
                // TODO: consider how to manage MxNDArray generated during inference
                allInputsNDArray[pair.getValue()] =
                        NDArray.create(this, new Shape(batchSize), device);
            }
        }
        NDArray[] result = JnaUtils.cachedOpInvoke(getParent(), getHandle(), allInputsNDArray);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!getClosed()) {
            logger.debug(String.format("Start to free CachedOp instance: %S", this.getUid()));
            super.freeSubResources();
            Pointer pointer = handle.getAndSet(null);
            if (pointer != null) {
                JnaUtils.freeCachedOp(pointer);
            }
            setClosed(true);
            logger.debug(String.format("Finish to free CachedOp instance: %S", this.getUid()));
        }
    }

    private int indexOf(String inputName, int position) {
        if (inputName == null) {
            return dataIndices.valueAt(position);
        }

        Integer index = dataIndicesMap.get(inputName);
        if (index == null) {
            throw new IllegalArgumentException(
                    "Unknown input name: "
                            + inputName
                            + ", expected inputs: "
                            + dataIndicesMap.keySet().toString());
        }
        return index;
    }
}
