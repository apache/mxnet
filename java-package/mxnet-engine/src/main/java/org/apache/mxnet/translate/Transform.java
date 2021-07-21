package org.apache.mxnet.translate;

import org.apache.mxnet.ndarray.MxNDArray;

/**
 * An interface to apply various transforms to the input.
 *
 * <p>A transform can be any function that modifies the input. Some examples of transform are crop
 * and resize.
 */
// TODO : not used by now
public interface Transform {
    /**
     * Applies the {@code Transform} to the given {@link MxNDArray}.
     *
     * @param array the {@link MxNDArray} on which the {@link Transform} is applied
     * @return the output of the {@code Transform}
     */
    MxNDArray transform(MxNDArray array);
}
