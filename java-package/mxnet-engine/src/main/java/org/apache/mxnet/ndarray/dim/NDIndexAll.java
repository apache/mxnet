package org.apache.mxnet.ndarray.dim;

/** An {@code NDIndexElement} to return all values in a particular dimension. */
public class NDIndexAll implements NDIndexElement {

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return 1;
    }
}