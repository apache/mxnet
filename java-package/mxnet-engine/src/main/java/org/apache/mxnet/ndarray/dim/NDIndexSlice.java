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
package org.apache.mxnet.ndarray.dim;

/** An NDIndexElement that returns a range of values in the specified dimension. */
public class NDIndexSlice implements NDIndexElement {

    private Long min;
    private Long max;
    private Long step;

    /**
     * Constructs a {@code NDIndexSlice} instance with specified range and step.
     *
     * @param min the start of the range
     * @param max the end of the range
     * @param step the step between each slice
     * @throws IllegalArgumentException Thrown if the step is zero
     */
    public NDIndexSlice(Long min, Long max, Long step) {
        this.min = min;
        this.max = max;
        this.step = step;
        if (step != null && step == 0) {
            throw new IllegalArgumentException("The step can not be zero");
        }
    }

    /**
     * Returns the start of the range.
     *
     * @return the start of the range
     */
    public Long getMin() {
        return min;
    }

    /**
     * Returns the end of the range.
     *
     * @return the end of the range
     */
    public Long getMax() {
        return max;
    }

    /**
     * Returns the step between each slice.
     *
     * @return the step between each slice
     */
    public Long getStep() {
        return step;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
