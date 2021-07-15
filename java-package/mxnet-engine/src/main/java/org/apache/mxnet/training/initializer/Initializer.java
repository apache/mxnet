package org.apache.mxnet.training.initializer;

import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link MxNDArray} parameters stored within a {@link Block}.
 *
 * @see <a
 *     href="https://d2l.djl.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html">The
 *     D2L chapter on numerical stability and initialization</a>
 */
public interface Initializer {

    Initializer ZEROS = (p, s, t, d) -> MxNDArray.create(p, s, t, d).zeros();
    Initializer ONES = (p, s, t, d) -> MxNDArray.create(p, s, t, d).ones();

    /**
     * Initializes a single {@link MxNDArray}.
     *
     * @param shape the {@link Shape} for the new NDArray
     * @param dataType the {@link DataType} for the new NDArray
     * @return the {@link MxNDArray} initialized with the manager and shape
     */
    MxNDArray initialize(MxResource parent, Shape shape, DataType dataType, Device device);
}