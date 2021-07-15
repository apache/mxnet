package org.apache.mxnet.ndarray;

import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.ndarray.dim.NDIndexBooleans;
import org.apache.mxnet.ndarray.dim.NDIndexElement;
import org.apache.mxnet.ndarray.dim.full.NDIndexFullPick;
import org.apache.mxnet.ndarray.dim.full.NDIndexFullSlice;
import org.apache.mxnet.ndarray.index.NDIndex;
import org.apache.mxnet.ndarray.types.Shape;

import java.util.List;
import java.util.Optional;
import java.util.Stack;

/** A helper class for {@link MxNDArray} implementations for operations with an {@link NDIndex}. */
public class NDArrayIndexer {

    public MxNDArray get(MxNDArray array, NDIndex index) {
        if (index.getRank() == 0 && array.getShape().isScalar()) {
            return array.duplicate();
        }

        // use booleanMask for NDIndexBooleans case
        List<NDIndexElement> indices = index.getIndices();
        if (!indices.isEmpty() && indices.get(0) instanceof NDIndexBooleans) {
            if (indices.size() != 1) {
                throw new IllegalArgumentException(
                        "get() currently didn't support more that one boolean NDArray");
            }
            return array.booleanMask(((NDIndexBooleans) indices.get(0)).getIndex());
        }

        Optional<NDIndexFullPick> fullPick = NDIndexFullPick.fromIndex(index, array.getShape());
        if (fullPick.isPresent()) {
            return get(array, fullPick.get());
        }

        Optional<NDIndexFullSlice> fullSlice = NDIndexFullSlice.fromIndex(index, array.getShape());
        if (fullSlice.isPresent()) {
            return get(array, fullSlice.get());
        }
        throw new UnsupportedOperationException(
                "get() currently supports all, fixed, and slices indices");
    }

    public MxNDArray get(MxNDArray array, NDIndexFullPick fullPick) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", fullPick.getAxis());
        params.addParam("keepdims", true);
        params.add("mode", "wrap");
        return MxNDArray.invoke(array.getParent(), "pick", new MxNDList(array, fullPick.getIndices()), params)
                .singletonOrThrow();
    }

    public MxNDArray get(MxNDArray array, NDIndexFullSlice fullSlice) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        MxNDArray result = MxNDArray.invoke(array.getParent(),"_npi_slice", array, params);
        int[] toSqueeze = fullSlice.getToSqueeze();
        if (toSqueeze.length > 0) {
            MxNDArray oldResult = result;
            result = result.squeeze(toSqueeze);
            oldResult.close();
        }
        return result;
    }

    public void set(MxNDArray array, NDIndexFullSlice fullSlice, MxNDArray value) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        Stack<MxNDArray> prepareValue = new Stack<>();
        prepareValue.add(value);
        prepareValue.add(prepareValue.peek().toDevice(array.getDevice(), false));
        // prepareValue.add(prepareValue.peek().asType(getDataType(), false));
        // Deal with the case target: (1, 10, 1), original (10)
        // try to find (10, 1) and reshape (10) to that
        Shape targetShape = fullSlice.getShape();
        while (targetShape.size() > value.size()) {
            targetShape = targetShape.slice(1);
        }
        prepareValue.add(prepareValue.peek().reshape(targetShape));
        prepareValue.add(prepareValue.peek().broadcast(fullSlice.getShape()));

        MxNDArray.invoke(
                        "_npi_slice_assign",
                        new MxNDArray[] {array, prepareValue.peek()},
                        new MxNDArray[] {array},
                        params);
        for (MxNDArray toClean : prepareValue) {
            if (toClean != value) {
                toClean.close();
            }
        }
    }

    public void set(MxNDArray array, NDIndexFullSlice fullSlice, Number value) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());
        params.addParam("scalar", value);
        MxNDArray.invoke(
                        "_npi_slice_assign_scalar",
                        new MxNDArray[] {array},
                        new MxNDArray[] {array},
                        params);
    }
}