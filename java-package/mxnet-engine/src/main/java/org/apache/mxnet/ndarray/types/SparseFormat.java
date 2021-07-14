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