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

package org.apache.mxnet.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.nn.Parameter;

/**
 * The {@code ParameterStore} contains a map from a parameter to the mirrors of it on other devices.
 */
public class ParameterStore extends MxResource {

    private Map<String, ParameterData> parameterMap;
    private Map<Device, Integer> deviceMap;
    private boolean copy;
    private ParameterServer parameterServer;

    /**
     * Constructs an empty {@code ParameterStore}.
     *
     * @param parent the parent {@link MxResource} to manage this instance
     * @param copy whether to always copy even for the same device as the original parameter
     */
    public ParameterStore(MxResource parent, boolean copy, Device device) {
        super();
        this.setParent(parent);
        this.copy = copy;
        parameterMap = new ConcurrentHashMap<>();
        deviceMap = new ConcurrentHashMap<>();
        deviceMap.put(device, 0);
    }

    /**
     * Sets the parameterServer used to apply updates to the parameters.
     *
     * @param parameterServer the parameterServer
     * @param devices the devices to create mirrored parameters on
     */
    public void setParameterServer(ParameterServer parameterServer, Device[] devices) {
        this.parameterServer = parameterServer;
        deviceMap.clear();
        for (int i = 0; i < devices.length; ++i) {
            if (deviceMap.put(devices[i], i) != null) {
                throw new IllegalArgumentException("Duplicated devices are not allowed.");
            }
        }
    }

    /** Updates all the mirrored parameters. */
    public void updateAllParameters() {
        for (Map.Entry<String, ParameterData> entry : parameterMap.entrySet()) {
            String parameterId = entry.getKey();
            ParameterData data = entry.getValue();
            if (data.requireGradient()) {
                NDArray[] params = data.toArray();
                parameterServer.update(parameterId, params);
            }
        }
    }

    /**
     * Returns the value of a mirrored parameter on a device.
     *
     * @param parameter the parameter to get the value for
     * @param device the device to get the mirror from
     * @param training true for a training forward pass
     * @return the value of the mirrored parameter on the device
     */
    public NDArray getValue(Parameter parameter, Device device, boolean training) {
        // for those optional parameters, they might not be in the ParameterStore
        if (parameter == null) {
            return null;
        }
        String parameterId = parameter.getId();
        int index = deviceMap.get(device);
        ParameterData data =
                parameterMap.computeIfAbsent(parameterId, k -> new ParameterData(parameter));

        if (data.isEmpty()) {
            NDArray array = parameter.getArray();

            if (parameterServer != null) {
                // initialize on parameter store for first time
                parameterServer.init(parameterId, new NDArray[] {array});
                NDArray[] arrays = new NDArray[deviceMap.size()];
                for (Map.Entry<Device, Integer> entry : deviceMap.entrySet()) {
                    Device dev = entry.getKey();
                    int i = entry.getValue();
                    if (i == index && array.getDevice().equals(dev)) {
                        arrays[i] = array;
                    } else {
                        arrays[i] = array.toDevice(dev, true);
                        //                        arrays[i].attach(manager);
                        // some parameter doesn't require grad
                        // for example running_mean in BatchNorm
                        if (parameter.requiresGradient()) {
                            arrays[i].setRequiresGradient(true);
                        }
                    }
                    data.add(arrays[i]);
                }
            } else {
                if (copy || !array.getDevice().equals(device)) {
                    array = array.toDevice(device, true);
                    //                    array.attach(manager);
                    // some parameter doesn't require grad
                    // for example running_mean in BatchNorm
                    if (parameter.requiresGradient() && training) {
                        array.setRequiresGradient(true);
                    }
                }
                data.add(array);
            }
        }

        return data.get(index);
    }

    /** Synchronizes the values on all mirrors with the main parameter. */
    public void sync() {
        for (ParameterData data : parameterMap.values()) {
            data.sync();
        }
    }

    /** A helper for {@link ParameterStore} that stores data for a single parameter. */
    private final class ParameterData {

        private Parameter parameter;
        private List<NDArray> list;

        private ParameterData(Parameter parameter) {
            this.parameter = parameter;
            list = Collections.synchronizedList(new ArrayList<>());
        }

        private List<NDArray> getNDArrays() {
            return list;
        }

        private boolean isEmpty() {
            return list.isEmpty();
        }

        private void add(NDArray array) {
            list.add(array);
        }

        private NDArray get(int index) {
            return list.get(index);
        }

        private NDArray[] toArray() {
            return list.toArray(new NDArray[0]);
        }

        private boolean requireGradient() {
            return parameter.requiresGradient();
        }

        private void sync() {
            NDArray array = parameter.getArray();
            Device device = array.getDevice();
            if (!deviceMap.containsKey(device)) {
                // model's parameters maybe loaded on different device than any of training devices.
                list.get(0).copyTo(array);
            }
        }
    }
}
