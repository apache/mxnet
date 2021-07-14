/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.apache.mxnet.ndarray.MxNDArray;

/** An {@link NDIndexElement} that gets elements by index in the specified axis. */
public class NDIndexPick implements NDIndexElement {

    private MxNDArray indices;

    /**
     * Constructs a pick.
     *
     * @param indices the indices to pick
     */
    public NDIndexPick(MxNDArray indices) {
        this.indices = indices;
    }

    @Override
    /** {@inheritDoc} */
    public int getRank() {
        return 1;
    }

    /**
     * Returns the indices to pick.
     *
     * @return the indices to pick
     */
    public MxNDArray getIndices() {
        return indices;
    }
}
