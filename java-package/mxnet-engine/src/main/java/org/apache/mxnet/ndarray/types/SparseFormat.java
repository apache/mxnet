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

package org.apache.mxnet.ndarray.types;

/**
 * An enum representing Sparse matrix storage formats.
 *
 * <ul>
 *   <li>DENSE: Stride format
 *   <li>ROW_SPARSE: Row Sparse
 *   <li>CSR: Compressed Sparse Row
 * </ul>
 *
 * @see <a href="https://software.intel.com/en-us/node/471374">Sparse Matrix Storage Formats</a>
 */
public enum SparseFormat {
    // the dense format is accelerated by MKLDNN by default
    DENSE("default", 0),
    ROW_SPARSE("row_sparse", 1),
    CSR("csr", 2),
    COO("coo", 3);

    private String type;
    private int value;

    SparseFormat(String type, int value) {
        this.type = type;
        this.value = value;
    }

    /**
     * Gets the {@code SparseFormat} from it's integer value.
     *
     * @param value the integer value of the {@code SparseFormat}
     * @return a {@code SparseFormat}
     */
    public static SparseFormat fromValue(int value) {
        for (SparseFormat t : values()) {
            if (value == t.getValue()) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unknown Sparse type: " + value);
    }

    /**
     * Returns the {@code SparseFormat} name.
     *
     * @return the {@code SparseFormat} name
     */
    public String getType() {
        return type;
    }

    /**
     * Returns the integer value of this {@code SparseFormat}.
     *
     * @return the integer value of this {@code SparseFormat}
     */
    public int getValue() {
        return value;
    }
}