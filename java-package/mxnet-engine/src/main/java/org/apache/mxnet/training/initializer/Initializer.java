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

package org.apache.mxnet.training.initializer;

import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.ndarray.NDArray;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.SymbolBlock;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link NDArray} parameters stored within a {@link SymbolBlock}.
 *
 * @see <a
 *     href="https://d2l.djl.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html">The
 *     D2L chapter on numerical stability and initialization</a>
 */
public interface Initializer {

    Initializer ZEROS = (p, s, t, d) -> NDArray.create(p, s, t, d).zeros();
    Initializer ONES = (p, s, t, d) -> NDArray.create(p, s, t, d).ones();

    /**
     * Initializes a single {@link NDArray}.
     *
     * @param shape the {@link Shape} for the new NDArray
     * @param dataType the {@link DataType} for the new NDArray
     * @return the {@link NDArray} initialized with the manager and shape
     */
    NDArray initialize(MxResource parent, Shape shape, DataType dataType, Device device);
}
