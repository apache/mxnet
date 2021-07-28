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

package org.apache.mxnet.translate;

import java.io.IOException;
import org.apache.mxnet.engine.Model;
import org.apache.mxnet.engine.Predictor;

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
     * @param model the model to translate for
     * @throws IOException if there is an error reading inputs for preparing the translator
     */
    default void prepare(Model model) throws IOException {}
}
