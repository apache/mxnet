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

import com.sun.jna.Native;
import java.io.File;
import java.lang.management.MemoryUsage;
import java.util.regex.Pattern;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.exception.JnaCallException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class containing CUDA utility methods. */
public final class CudaUtils {

    private static final Logger logger = LoggerFactory.getLogger(CudaUtils.class);

    private static final CudaLibrary LIB = loadLibrary();

    private static int gpuCount = -1;

    private CudaUtils() {}

    /**
     * Gets whether CUDA runtime library is in the system.
     *
     * @return {@code true} if CUDA runtime library is in the system
     */
    public static boolean hasCuda() {
        return getGpuCount() > 0;
    }

    /**
     * Returns the number of GPUs available in the system.
     *
     * @return the number of GPUs available in the system
     */
    public static int getGpuCount() {

        if (gpuCount != -1) {
            return gpuCount;
        }

        try {
            validateLibrary();
        } catch (IllegalStateException e) {
            return 0;
        }
        int[] count = new int[1];
        int result = LIB.cudaGetDeviceCount(count);
        switch (result) {
            case 0:
                gpuCount = count[0];
                return gpuCount;
            case CudaLibrary.ERROR_NO_DEVICE:
                logger.debug(
                        "No GPU device found: {} ({})", LIB.cudaGetErrorString(result), result);
                gpuCount = 0;
                return gpuCount;
            case CudaLibrary.INITIALIZATION_ERROR:
            case CudaLibrary.INSUFFICIENT_DRIVER:
            case CudaLibrary.ERROR_NOT_PERMITTED:
            default:
                logger.warn(
                        "Failed to detect GPU count: {} ({})",
                        LIB.cudaGetErrorString(result),
                        result);
                gpuCount = 0;
                return gpuCount;
        }
    }

    /**
     * Returns the version of CUDA runtime.
     *
     * @return the version if CUDA runtime
     */
    public static int getCudaVersion() {
        validateLibrary();
        int[] version = new int[1];
        int result = LIB.cudaRuntimeGetVersion(version);
        checkCall(result);
        return version[0];
    }

    /**
     * Returns the version string of CUDA runtime.
     *
     * @return the version string of CUDA runtime
     */
    public static String getCudaVersionString() {
        validateLibrary();
        int version = getCudaVersion();
        int major = version / 1000;
        int minor = (version / 10) % 10;
        return String.valueOf(major) + minor;
    }

    /**
     * Returns the CUDA compute capability.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return the CUDA compute capability
     */
    public static String getComputeCapability(int device) {
        validateLibrary();
        int attrComputeCapabilityMajor = 75;
        int attrComputeCapabilityMinor = 76;

        int[] major = new int[1];
        int[] minor = new int[1];
        checkCall(LIB.cudaDeviceGetAttribute(major, attrComputeCapabilityMajor, device));
        checkCall(LIB.cudaDeviceGetAttribute(minor, attrComputeCapabilityMinor, device));

        return String.valueOf(major[0] + minor[0]);
    }

    /**
     * Returns the {@link MemoryUsage} of the specified GPU device.
     *
     * @param device the GPU {@link Device} to retrieve
     * @return the {@link MemoryUsage} of the specified GPU device
     * @throws IllegalArgumentException if {@link Device} is not GPU device or does not exist
     */
    public static MemoryUsage getGpuMemory(Device device) {
        if (!Device.Type.GPU.equals(device.getDeviceType())) {
            throw new IllegalArgumentException("Only GPU device is allowed.");
        }

        validateLibrary("No GPU device detected.");

        int[] currentDevice = new int[1];
        checkCall(LIB.cudaGetDevice(currentDevice));
        checkCall(LIB.cudaSetDevice(device.getDeviceId()));

        long[] free = new long[1];
        long[] total = new long[1];

        checkCall(LIB.cudaMemGetInfo(free, total));
        checkCall(LIB.cudaSetDevice(currentDevice[0]));

        long committed = total[0] - free[0];
        return new MemoryUsage(-1, committed, committed, total[0]);
    }

    private static CudaLibrary loadLibrary() {
        try {
            if (System.getProperty("os.name").startsWith("Win")) {
                String path = System.getenv("PATH");
                if (path == null) {
                    return null;
                }
                Pattern p = Pattern.compile("cudart64_\\d+\\.ddl");
                String cudaPath = System.getenv("CUDA_PATH");

                String[] searchPath = getPathArray(path, cudaPath);

                for (String item : searchPath) {
                    File dir = new File(item);
                    File[] files = dir.listFiles(n -> p.matcher(n.getName()).matches());
                    if (files != null && files.length > 0) {
                        String fileName = files[0].getName();
                        String cudaRT = fileName.substring(0, fileName.length() - 4);
                        logger.debug("Found cudart: {}", files[0].getAbsolutePath());
                        return Native.load(cudaRT, CudaLibrary.class);
                    }
                }
                logger.debug("No cudart library found in path.");
                return null;
            }
            return Native.load("cudart", CudaLibrary.class);
        } catch (UnsatisfiedLinkError e) {
            logger.debug("cudart library not found.");
            logger.trace("", e);
            return null;
        }
    }

    private static String[] getPathArray(String path, String cudaPath) {
        if (cudaPath == null) {
            return path.split(";");
        } else {
            return ";".split(String.format("%s\\bin\\;%s", cudaPath, path));
        }
    }

    private static void checkCall(int ret) {
        validateLibrary();
        if (ret != 0) {
            throw new JnaCallException(
                    String.format(
                            "CUDA API call failed: %s (%d)", LIB.cudaGetErrorString(ret), ret));
        }
    }

    private static void validateLibrary() {
        if (LIB == null) {
            throw new IllegalStateException("No cuda library is loaded.");
        }
    }

    private static void validateLibrary(String msg) {
        if (msg == null) {
            validateLibrary();
        } else if (LIB == null) {
            throw new IllegalStateException(msg);
        }
    }
}
