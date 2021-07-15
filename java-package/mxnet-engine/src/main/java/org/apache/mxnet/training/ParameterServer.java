package org.apache.mxnet.training;

import org.apache.mxnet.ndarray.MxNDArray;

import java.util.Arrays;

/** An interface for a key-value store to store parameters, and their corresponding gradients. */
public interface ParameterServer extends AutoCloseable {

    /**
     * Initializes the {@code ParameterStore} for the given parameter.
     *
     * @param parameterId the parameter ID
     * @param value the values to be set for the given parameter
     */
    void init(String parameterId, MxNDArray[] value);

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param params the parameter NDArrays in different devices to be updated.
     */
    default void update(String parameterId, MxNDArray[] params) {
        MxNDArray[] grads = Arrays.stream(params).map(MxNDArray::getGradient).toArray(MxNDArray[]::new);
        update(parameterId, grads, params);
        Arrays.stream(grads).forEach(MxNDArray::close);
    }
    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param grads the gradient NDArrays in different devices to apply the update.
     * @param params the parameter NDArrays in different devices to be updated.
     */
    void update(String parameterId, MxNDArray[] grads, MxNDArray[] params);

    /** {@inheritDoc} */
    @Override
    void close();
}