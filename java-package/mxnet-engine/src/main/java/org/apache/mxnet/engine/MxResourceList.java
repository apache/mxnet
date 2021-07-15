package org.apache.mxnet.engine;

import org.apache.mxnet.util.Pair;
import org.apache.mxnet.util.PairList;

import java.util.List;
import java.util.Map;

public class MxResourceList extends PairList<String, MxResource> {

    /** Creates an empty {@code MxResourceList}. */
    public MxResourceList() {}

    /**
     * Constructs an empty {@code MxResourceList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public MxResourceList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code MxResourceList}
     * @param values the value list containing the elements to be placed into this {@code MxResource}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public MxResourceList(List<String> keys, List<MxResource> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code MxResourceList}
     */
    public MxResourceList(List<Pair<String, MxResource>> list) {
        super(list);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public MxResourceList(Map<String, MxResource> map) {
        super(map);
    }

}
