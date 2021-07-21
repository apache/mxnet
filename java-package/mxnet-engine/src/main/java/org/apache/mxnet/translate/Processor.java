package org.apache.mxnet.translate;

import org.apache.mxnet.ndarray.MxNDList;

/**
 * An interface that provides pre-processing and post-processing functionality.
 *
 * @param <I> the type of the input object
 */
public interface Processor<I, O> {

    /**
     * Gets the {@link Pipeline} applied to the input.
     *
     * @return the {@link Pipeline}
     */
    default Pipeline getPipeline() {
        throw new UnsupportedOperationException("Not implemented.");
    }

    /**
     * Processes the input and converts it to NDList.
     *
     * @param input the input object
     * @return the {@link MxNDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    MxNDList processInput(I input) throws Exception;

    /**
     * Processes the input and converts it to NDList.
     *
     * @param output the input object
     * @return the {@link MxNDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    O processOutput(MxNDList output) throws Exception;

}
