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

package org.apache.mxnet.util.cuda;

import com.sun.jna.Library;

/**
 * {@code CudaLibrary} contains methods mapping to CUDA runtime API.
 *
 * <p>see: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
 */
public interface CudaLibrary extends Library {

    int INITIALIZATION_ERROR = 3;
    int INSUFFICIENT_DRIVER = 35;
    int ERROR_NO_DEVICE = 100;
    int ERROR_NOT_PERMITTED = 800;

    /**
     * Gets the number of devices with compute capability greater or equal to 1.0 that are available
     * for execution.
     *
     * @param deviceCount the returned device count
     * @return CUDA runtime API error code
     */
    int cudaGetDeviceCount(int[] deviceCount);

    /**
     * Returns the version number of the installed CUDA Runtime.
     *
     * @param runtimeVersion output buffer of runtime version number
     * @return CUDA runtime API error code
     */
    int cudaRuntimeGetVersion(int[] runtimeVersion);

    /**
     * Gets the integer value of the attribute {@code attr} on device.
     *
     * @param pi the returned device attribute value
     * @param attr the device attribute to query
     * @param device the GPU device to retrieve
     * @return CUDA runtime API error code
     */
    int cudaDeviceGetAttribute(int[] pi, int attr, int device);

    /**
     * Gets free and total device memory.
     *
     * @param free the returned free memory in bytes
     * @param total the returned total memory in bytes
     * @return CUDA runtime API error code
     */
    int cudaMemGetInfo(long[] free, long[] total);

    /**
     * Set device to be used for GPU executions.
     *
     * @param device the GPU device to retrieve
     * @return CUDA runtime API error code
     */
    int cudaSetDevice(int device);

    /**
     * Gets which device is currently being used.
     *
     * @param device the returned current device
     * @return CUDA runtime API error code
     */
    int cudaGetDevice(int[] device);

    /**
     * Returns the description string for an error code.
     *
     * @param code the CUDA error code to convert to string
     * @return the description string for an error code
     */
    String cudaGetErrorString(int code);
}
