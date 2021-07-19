package org.apache.mxnet.nn;

import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

import java.util.List;
import java.util.Map;

/** Represents a set of names and Parameters. */
public class ParameterList extends PairList<String, Parameter> {

    /** Create an empty {@code ParameterList}. */
    public ParameterList() {}

    /**
     * Constructs an empty {@code ParameterList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public ParameterList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code ParameterList}
     * @param values the value list containing the elements to be placed into this {@code
     *     ParameterList}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public ParameterList(List<String> keys, List<Parameter> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code ParameterList}
     */
    public ParameterList(List<Pair<String, Parameter>> list) {
        super(list);
    }

    /**
     * Constructs a {@code ParameterList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public ParameterList(Map<String, Parameter> map) {
        super(map);
    }
}
