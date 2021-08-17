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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.exception.TranslateException;
import org.apache.mxnet.ndarray.NDList;
import org.apache.mxnet.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Predictor} class provides a session for model inference.
 *
 * <p>You can use a {@code Predictor}, with a specified {@link Translator}, to perform inference on
 * a {@link Model}
 *
 * @param <I> the input type
 * @param <O> the output type
 * @see Model
 * @see Translator
 * @see <a href="http://docs.djl.ai/docs/development/memory_management.html">The guide on memory
 *     management</a>
 * @see <a
 *     href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/multithread_inference.md">The
 *     guide on running multi-threaded inference</a>
 * @see <a href="http://docs.djl.ai/docs/development/inference_performance_optimization.html">The
 *     guide on inference performance optimization</a>
 */
public class Predictor<I, O> extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(Predictor.class);
    private Translator<I, O> translator;
    private Model model;

    /**
     * Creates a new instance of {@code Predictor} with the given {@link Model} and {@link
     * Translator}.
     *
     * @param model the model on which the predictions are based
     * @param translator the translator to be used
     */
    public Predictor(Model model, Translator<I, O> translator) {
        super(model);
        this.model = model;
        this.translator = translator;
    }

    /**
     * Predicts an item for inference.
     *
     * @param input the input
     * @return the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    @SuppressWarnings("PMD.AvoidRethrowingException")
    public List<O> predict(List<I> input) {
        NDList[] ndLists = processInputs(input);
        for (int i = 0; i < ndLists.length; ++i) {
            ndLists[i] = forward(ndLists[i]);
        }
        return processOutPut(ndLists);
    }

    /**
     * Predicts an Item for inference.
     *
     * @param input input data
     * @return O the output object defined by the user
     * @throws TranslateException if an error occurs during prediction
     */
    public O predict(I input) {
        return predict(Collections.singletonList(input)).get(0);
    }

    private NDList forward(NDList ndList) {
        logger.trace("Predictor input data: {}", ndList);
        return model.getSymbolBlock().forward(ndList);
    }

    // TODO: add batch predict

    private NDList[] processInputs(List<I> inputs) {
        int batchSize = inputs.size();
        NDList[] preprocessed = new NDList[batchSize];
        try {
            for (int i = 0; i < batchSize; ++i) {
                preprocessed[i] = translator.processInput(inputs.get(i));
            }
        } catch (Exception e) {
            logger.error("Error occurs when process input items.", e);
            throw new TranslateException(e);
        }
        return preprocessed;
    }

    private List<O> processOutPut(NDList[] ndLists) {
        List<O> outputs = new ArrayList<>();
        try {
            for (NDList mxNDList : ndLists) {
                outputs.add(translator.processOutput(mxNDList));
            }
        } catch (Exception e) {
            logger.error("Error occurs when process output items.", e);
            throw new TranslateException(e);
        }
        return outputs;
    }
}
