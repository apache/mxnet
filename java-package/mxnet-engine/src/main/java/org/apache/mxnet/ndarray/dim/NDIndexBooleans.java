package org.apache.mxnet.ndarray.dim;

import org.apache.mxnet.ndarray.MxNDArray;

/** An {@code NDIndexElement} to return values based on a mask binary NDArray. */
public class NDIndexBooleans implements NDIndexElement {

    private MxNDArray index;

    /**
     * Constructs a {@code NDIndexBooleans} instance with specified mask binary NDArray.
     *
     * @param index the mask binary {@code NDArray}
     */
    public NDIndexBooleans(MxNDArray index) {
        this.index = index;
    }

    /**
     * Returns the mask binary {@code NDArray}.
     *
     * @return the mask binary {@code NDArray}
     */
    public MxNDArray getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return index.getShape().dimension();
    }
}