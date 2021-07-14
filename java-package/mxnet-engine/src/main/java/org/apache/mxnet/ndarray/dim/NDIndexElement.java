package org.apache.mxnet.ndarray.dim;

/** An index for particular dimensions created by NDIndex. */
public interface NDIndexElement {

    /**
     * Returns the number of dimensions occupied by this index element.
     *
     * @return the number of dimensions occupied by this index element
     */
    int getRank();
}
