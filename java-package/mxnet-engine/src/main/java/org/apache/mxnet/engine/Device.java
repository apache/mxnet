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
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.util.cuda.CudaUtils;

/**
 * The {@code Device} class provides the specified assignment for CPU/GPU processing on the {@link
 * org.apache.mxnet.ndarray.NDArray}.
 *
 * <p>Users can use this to specify whether to load/compute the {@code NDArray} on CPU/GPU with
 * deviceType and deviceId provided.
 */
public final class Device {

    private static final Map<String, Device> CACHE = new ConcurrentHashMap<>();

    private static final Device CPU = new Device(Type.CPU, -1);

    private static final Device GPU = Device.of(Type.GPU, 0);

    private String deviceType;

    private int deviceId;

    private static final Device DEFAULT_DEVICE = CPU;

    /**
     * Creates a {@code Device} with basic information.
     *
     * @param deviceType the device type, typically CPU or GPU
     * @param deviceId the deviceId on the hardware. For example, if you have multiple GPUs, you can
     *     choose which GPU to process the NDArray
     */
    private Device(String deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
    }

    /**
     * Returns a {@code Device} with device type and device id.
     *
     * @param deviceType the device type, typically CPU or GPU
     * @param deviceId the deviceId on the hardware.
     * @return a {@code Device} instance
     */
    public static Device of(String deviceType, int deviceId) {
        if (Type.CPU.equals(deviceType)) {
            return CPU;
        }
        String key = deviceType + '-' + deviceId;
        return CACHE.computeIfAbsent(key, k -> new Device(deviceType, deviceId));
    }

    /**
     * Returns the device type of the Device.
     *
     * @return the device type of the Device
     */
    public String getDeviceType() {
        return deviceType;
    }

    /**
     * Returns the {@code deviceId} of the Device.
     *
     * @return the {@code deviceId} of the Device
     */
    public int getDeviceId() {
        if (Type.CPU.equals(deviceType)) {
            throw new IllegalStateException("CPU doesn't have device id");
        }
        return deviceId;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (Type.CPU.equals(deviceType)) {
            return deviceType + "()";
        }
        return deviceType + '(' + deviceId + ')';
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Device device = (Device) o;
        if (Type.CPU.equals(deviceType)) {
            return Objects.equals(deviceType, device.getDeviceType());
        }
        return deviceId == device.deviceId && Objects.equals(deviceType, device.deviceType);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(deviceType, deviceId);
    }

    /**
     * Returns the default CPU Device.
     *
     * @return the default CPU Device
     */
    public static Device cpu() {
        return CPU;
    }

    /**
     * Returns the default GPU Device.
     *
     * @return the default GPU Device
     */
    public static Device gpu() {
        return GPU;
    }

    /**
     * Returns a new instance of GPU {@code Device} with the specified {@code deviceId}.
     *
     * @param deviceId the GPU device ID
     * @return a new instance of GPU {@code Device} with specified {@code deviceId}
     */
    public static Device gpu(int deviceId) {
        return of(Type.GPU, deviceId);
    }

    /**
     * Returns an array of devices.
     *
     * <p>If GPUs are available, it will return an array of {@code Device} of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @return an array of devices
     */
    public static Device[] getDevices() {
        return getDevices(Integer.MAX_VALUE);
    }

    /**
     * Returns an array of devices given the maximum number of GPUs to use.
     *
     * <p>If GPUs are available, it will return an array of {@code Device} of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @param maxGpus the max number of GPUs to use. Use 0 for no GPUs.
     * @return an array of devices
     */
    public static Device[] getDevices(int maxGpus) {
        int count = getGpuCount();
        if (maxGpus <= 0 || count <= 0) {
            return new Device[] {CPU};
        }

        count = Math.min(maxGpus, count);
        Device[] devices = new Device[count];
        for (int i = 0; i < devices.length; ++i) {
            devices[i] = gpu(i);
        }
        return devices;
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return the number of GPUs available in the system
     */
    public static int getGpuCount() {
        return CudaUtils.getGpuCount();
    }

    /**
     * Returns the default context used in Engine.
     *
     * <p>The default type is defined by whether the deep learning engine is recognizing GPUs
     * available on your machine. If there is no GPU available, CPU will be used.
     *
     * @return a {@link Device}
     */
    private static Device defaultDevice() {
        return DEFAULT_DEVICE;
    }

    /**
     * Returns the given device or the default if it is null.
     *
     * @param device the device to try to return
     * @return the given device or the default if it is null
     */
    public static Device defaultIfNull(Device device) {
        if (device != null) {
            return device;
        }
        return defaultDevice();
    }

    /**
     * Returns the default device.
     *
     * @return the default device
     */
    public static Device defaultIfNull() {
        return defaultIfNull(null);
    }

    /** Contains device type string constants. */
    public interface Type {
        String CPU = "cpu";
        String GPU = "gpu";
    }
}
