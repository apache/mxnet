package org.apache.mxnet.translate;

import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.Predictor;

import java.io.IOException;

/**
 * The {@code Translator} interface provides model pre-processing and postprocessing functionality.
 *
 * <p>Users can use this in {@link Predictor} with input and output objects specified. The following
 * is an example of processing an image and creating classification output:
 *
 * @param <I> the input type
 * @param <O> the output type
 */
public interface Translator<I, O> extends Processor<I, O> {
    // TODO: implement getPipeline() and related methods
    /**
     * Prepares the translator with the manager and model to use.
     *
     * @param mxModel the model to translate for
     * @throws IOException if there is an error reading inputs for preparing the translator
     */
    default void prepare(MxModel mxModel) throws IOException {}
}
