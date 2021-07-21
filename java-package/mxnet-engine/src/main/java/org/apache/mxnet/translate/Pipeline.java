package org.apache.mxnet.translate;

import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** {@code Pipeline} allows applying multiple transforms on an input {@link MxNDList}. */
public class Pipeline {

    private PairList<IndexKey, Transform> transforms;

    /** Creates a new instance of {@code Pipeline} that has no {@link Transform} defined yet. */
    public Pipeline() {
        transforms = new PairList<>();
    }

    /**
     * Creates a new instance of {@code Pipeline} that can apply the given transforms on its input.
     *
     * <p>Since no keys are provided for these transforms, they will be applied to the first element
     * in the input {@link MxNDList} when the {@link #transform(MxNDList) transform} method is called on
     * this object.
     *
     * @param transforms the transforms to be applied when the {@link #transform(MxNDList) transform}
     *     method is called on this object
     */
    public Pipeline(Transform... transforms) {
        this.transforms = new PairList<>();
        for (Transform transform : transforms) {
            this.transforms.add(new IndexKey(0), transform);
        }
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the input when
     * the {@link #transform(MxNDList) transform} method is called on this object.
     *
     * <p>Since no keys are provided for this {@link Transform}, it will be applied to the first
     * element in the input {@link MxNDList}.
     *
     * @param transform the {@link Transform} to be added
     * @return this {@code Pipeline}
     */
    public Pipeline add(Transform transform) {
        transforms.add(new IndexKey(0), transform);
        return this;
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the {@link
     * MxNDArray} at the given index in the input {@link MxNDList}.
     *
     * @param index the index corresponding to the {@link MxNDArray} in the input {@link MxNDList} on
     *     which the given transform must be applied to
     * @param transform the {@link Transform} to be added
     * @return this {@code Pipeline}
     */
    public Pipeline add(int index, Transform transform) {
        transforms.add(new IndexKey(index), transform);
        return this;
    }

    /**
     * Adds the given {@link Transform} to the list of transforms to be applied on the {@link
     * MxNDArray} with the given key as name in the input {@link MxNDList}.
     *
     * @param name the key corresponding to the {@link MxNDArray} in the input {@link MxNDList} on which
     *     the given transform must be applied to
     * @param transform the {@code Transform} to be applied when the {@link #transform(MxNDList)
     *     transform} method is called on this object
     * @return this {@code Pipeline}
     */
    public Pipeline add(String name, Transform transform) {
        transforms.add(new IndexKey(name), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position.
     *
     * <p>Since no keys or indices are provided for this {@link Transform}, it will be applied to
     * the first element in the input {@link MxNDList} when the {@link #transform(MxNDList) transform}
     * method is called on this object.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, Transform transform) {
        transforms.add(position, new IndexKey(0), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position to be
     * applied on the {@link MxNDArray} at the given index in the input {@link MxNDList}.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param index the index corresponding to the {@link MxNDArray} in the input {@link MxNDList} on
     *     which the given transform must be applied to
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, int index, Transform transform) {
        transforms.add(position, new IndexKey(index), transform);
        return this;
    }

    /**
     * Inserts the given {@link Transform} to the list of transforms at the given position to be
     * applied on the {@link MxNDArray} with the given name in the input {@link MxNDList}.
     *
     * @param position the position at which the {@link Transform} must be inserted
     * @param name the key corresponding to the {@link MxNDArray} in the input {@link MxNDList} on which
     *     the given transform must be applied to
     * @param transform the {@code Transform} to be inserted
     * @return this {@code Pipeline}
     */
    public Pipeline insert(int position, String name, Transform transform) {
        transforms.add(position, new IndexKey(name), transform);
        return this;
    }

    /**
     * Applies the transforms configured in this object on the input {@link MxNDList}.
     *
     * <p>If a key is specified with the transform, those transforms will only be applied to the
     * {@link MxNDArray} in the input {@link MxNDList}. If a key is not specified, it will be applied to
     * the first element in the input {@link MxNDList}.
     *
     * @param input the input {@link MxNDList} on which the tranforms are to be applied
     * @return the output {@link MxNDList} after applying the tranforms
     */
    public MxNDList transform(MxNDList input) {
        if (transforms.isEmpty() || input.isEmpty()) {
            return input;
        }

        MxNDArray[] arrays = input.toArray(new MxNDArray[0]);

        Map<IndexKey, Integer> map = new ConcurrentHashMap<>();
        // create mapping
        for (int i = 0; i < input.size(); i++) {
            String key = input.get(i).getName();
            if (key != null) {
                map.put(new IndexKey(key), i);
            }
            map.put(new IndexKey(i), i);
        }
        // apply transform
        for (Pair<IndexKey, Transform> transform : transforms) {
            IndexKey key = transform.getKey();
            int index = map.get(key);
            MxNDArray array = arrays[index];

            arrays[index] = transform.getValue().transform(array);
            arrays[index].setName(array.getName());
        }

        return new MxNDList(arrays);
    }

    private static final class IndexKey {
        private String key;
        private int index;

        private IndexKey(String key) {
            this.key = key;
        }

        private IndexKey(int index) {
            this.index = index;
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            if (key == null) {
                return index;
            }
            return key.hashCode();
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof IndexKey)) {
                return false;
            }
            IndexKey other = (IndexKey) obj;
            if (key == null) {
                return index == other.index;
            }
            return key.equals(other.key);
        }
    }
}
