package org.apache.mxnet.ndarray.types;

import com.sun.jna.Pointer;
import org.apache.mxnet.ndarray.MxNDArray;

/**
 * An interface representing a Sparse NDArray.
 *
 * @see SparseFormat
 * @see <a href="https://software.intel.com/en-us/node/471374">Sparse Matrix Storage Formats</a>
 */
public class SparseNDArray extends MxNDArray {
    protected SparseNDArray(Pointer handle) {
        super(handle);
    }
}
