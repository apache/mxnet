package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * The {@code CachedOp} is an internal helper that provides the core functionality to execute a
 * {@link MxSymbolBlock}.
 *
 * <p>We don't recommended users interact with this class directly. Users should use {@link
 * Predictor} instead. CachedOp is an operator that simplifies calling and
 * analyzing the input shape. It requires minimum input to do inference because most of the
 * information can be obtained from the model itself.
 */
public class CachedOp extends MxResource {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private List<Parameter> parameters;
    private PairList<String, Integer> dataIndices;
    private Map<String, Integer> dataIndicesMap;
    private List<Integer> paramIndices;

    /**
     * Creates an instance of {@link CachedOp}.
     *
     * <p>It can be created by using {@link JnaUtils#createCachedOp(MxSymbolBlock, MxResource, boolean)}
     *
     * @param parent the MxResource object to manage this instance of CachedOp
     * @param handle the C handle of the CachedOp
     * @param parameters the parameter values
     * @param paramIndices the parameters required by the model and their corresponding location
     * @param dataIndices the input data names required by the model and their corresponding
     *     location
     */
    public CachedOp(
            MxResource parent,
            Pointer handle,
            List<Parameter> parameters,
            List<Integer> paramIndices,
            PairList<String, Integer> dataIndices) {
        super(parent, handle);
        this.parameters = parameters;
        this.dataIndices = dataIndices;
        this.paramIndices = paramIndices;
        this.dataIndicesMap = dataIndices.toMap();
    }

    /**
     * Assigns inputs to the empty locations of the input NDArray.
     *
     * @param parameterStore the parameterStore
     * @param data the input in {@link MxNDList} format
     * @param training true for a training forward pass
     * @return an {@link MxNDList}
     */
    public MxNDList forward(ParameterStore parameterStore, MxNDList data, boolean training) {
        // reset the input data index at the beginning
        MxNDArray[] allInputsNDArray = new MxNDArray[parameters.size()];
        // check device of input
        Device device = data.isEmpty() ? Device.defaultIfNull() : data.head().getDevice();
        // get the manager of the data
        // fill allInputsNDArray with parameter values on correct device
        for (int index : paramIndices) {
            Parameter parameter = parameters.get(index);
            MxNDArray value = parameterStore.getValue(parameter, device, training);
            if (value == null) {
                throw new NullPointerException("Failed to find parameter from parameterStore");
            }
            allInputsNDArray[index] = value;
        }

        // fill allInputsNDArray with data values
        int index = 0;
        for (MxNDArray array : data) {
            // TODO: NDArray name doesn't match. To confirm the format of input name
//            String inputName = array.getName().split(":")[1];
            String inputName = array.getName();
            // if inputName not provided, value will follow the default order
            int idx = indexOf(inputName, index++);
            allInputsNDArray[idx] = array;
        }

        // check the input, set as Shape(batchSize) by default
        for (Pair<String, Integer> pair : dataIndices) {
            if (allInputsNDArray[pair.getValue()] == null) {
                // TODO: Do we need to set default to the input?
                long batchSize = data.head().getShape().get(0);
                String key = pair.getKey();
                if (!"prob_label".equals(key) && !"softmax_label".equals(key)) {
                    logger.warn(
                            "Input "
                                    + key
                                    + " not found, set NDArray to Shape("
                                    + batchSize
                                    + ") by default");
                }
                allInputsNDArray[pair.getValue()] = MxNDArray.create(this, new Shape(batchSize), device);
            }
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(getParent(), getHandle(), allInputsNDArray);
        return new MxNDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (!getClosed()) {
            Pointer pointer = handle.getAndSet(null);
            if (pointer != null) {
                JnaUtils.freeCachedOp(pointer);
            }
            setClosed();
        }
    }

    private int indexOf(String inputName, int position) {
        if (inputName == null) {
            return dataIndices.valueAt(position);
        }

        Integer index = dataIndicesMap.get(inputName);
        if (index == null) {
            throw new IllegalArgumentException(
                    "Unknown input name: "
                            + inputName
                            + ", expected inputs: "
                            + dataIndicesMap.keySet().toString());
        }
        return index;
    }

}