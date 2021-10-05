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

/** {@code DeviceType} is a class used to map the Device name to their corresponding type number. */
public final class DeviceType {

    private static final String CPU_PINNED = "cpu_pinned";

    private DeviceType() {}

    /**
     * Converts a {@link Device} to the corresponding MXNet device number.
     *
     * @param device the java {@link Device}
     * @return the MXNet device number
     * @exception IllegalArgumentException the device is null or is not supported
     */
    public static int toDeviceType(Device device) {
        if (device == null) {
            throw new IllegalArgumentException("Unsupported device: null");
        }

        String deviceType = device.getDeviceType();

        if (Device.Type.CPU.equals(deviceType)) {
            return 1;
        } else if (Device.Type.GPU.equals(deviceType)) {
            return 2;
        } else if (CPU_PINNED.equals(deviceType)) {
            return 3;
        } else {
            throw new IllegalArgumentException("Unsupported device: " + device.toString());
        }
    }

    /**
     * Converts from an MXNet device number to {@link Device}.
     *
     * @param deviceType the MXNet device number
     * @return the corresponding {@link Device}
     */
    public static String fromDeviceType(int deviceType) {
        switch (deviceType) {
            case 1:
            case 3:
                // hide the CPU_PINNED to frontend user
                // but the advance user can still create CPU_PINNED
                // to pass through engine
                return Device.Type.CPU;
            case 2:
                return Device.Type.GPU;
            default:
                throw new IllegalArgumentException("Unsupported deviceType: " + deviceType);
        }
    }
}
