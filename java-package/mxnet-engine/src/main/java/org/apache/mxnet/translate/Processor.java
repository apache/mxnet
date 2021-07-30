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

import org.apache.mxnet.ndarray.NDList;

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
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    NDList processInput(I input) throws Exception;

    /**
     * Processes the input and converts it to NDList.
     *
     * @param output the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    O processOutput(NDList output) throws Exception;
}
