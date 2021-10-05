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
 * Default no operational implement for {@link Translator} to process input and output {@link
 * org.apache.mxnet.ndarray.NDArray}.
 */
public class NoOpTranslator implements Translator<NDList, NDList> {

    /** {@inheritDoc} */
    @Override
    public Pipeline getPipeline() {
        return Translator.super.getPipeline();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(NDList input) {
        return input;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processOutput(NDList output) {
        return output;
    }
}
