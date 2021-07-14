/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.apache.mxnet.ndarray.dim;

/** An NDIndexElement that returns only a specific value in the corresponding dimension. */
public class NDIndexFixed implements NDIndexElement {

    private long index;

    /**
     * Constructs a {@code NDIndexFixed} instance with specified dimension.
     *
     * @param index the dimension of the NDArray
     */
    public NDIndexFixed(long index) {
        this.index = index;
    }

    /**
     * Returns the dimension of the index.
     *
     * @return the dimension of the index
     */
    public long getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}
