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
