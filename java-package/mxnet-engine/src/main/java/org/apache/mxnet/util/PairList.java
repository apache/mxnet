package org.apache.mxnet.util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The {@code PairList} class provides an efficient way to access a list of key-value pairs.
 *
 * @param <K> the key type
 * @param <V> the value type
 */
public class PairList<K, V> implements Iterable<Pair<K, V>> {

    private List<K> keys;
    private List<V> values;

    /** Constructs an empty {@code PairList}. */
    public PairList() {
        keys = new ArrayList<>();
        values = new ArrayList<>();
    }

    /**
     * Constructs an empty {@code PairList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public PairList(int initialCapacity) {
        keys = new ArrayList<>(initialCapacity);
        values = new ArrayList<>(initialCapacity);
    }

    /**
     * Constructs a {@code PairList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing elements to be placed into this PairList
     * @param values the value list containing elements to be placed into this PairList
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public PairList(List<K> keys, List<V> values) {
        if (keys.size() != values.size()) {
            throw new IllegalArgumentException("key value size mismatch.");
        }
        this.keys = keys;
        this.values = values;
    }

    /**
     * Constructs a {@code PairList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing elements to be placed into this PairList
     */
    public PairList(List<Pair<K, V>> list) {
        this(list.size());
        for (Pair<K, V> pair : list) {
            keys.add(pair.getKey());
            values.add(pair.getValue());
        }
    }

    /**
     * Constructs a {@code PairList} containing the elements of the specified map.
     *
     * @param map the map contains keys and values
     */
    public PairList(Map<K, V> map) {
        keys = new ArrayList<>(map.size());
        values = new ArrayList<>(map.size());
        for (Map.Entry<K, V> entry : map.entrySet()) {
            keys.add(entry.getKey());
            values.add(entry.getValue());
        }
    }

    /**
     * Inserts the specified element at the specified position in this list (optional operation),
     * and shifts the element currently at that position (if any) and any subsequent elements to the
     * right (adds one to their indices).
     *
     * @param index the index at which the specified element is to be inserted
     * @param key the key
     * @param value the value
     */
    public void add(int index, K key, V value) {
        keys.add(index, key);
        values.add(index, value);
    }

    /**
     * Adds a key and value to the list.
     *
     * @param key the key
     * @param value the value
     */
    public void add(K key, V value) {
        keys.add(key);
        values.add(value);
    }

    /**
     * Appends all of the elements in the specified pair list to the end of this list.
     *
     * @param other the {@code PairList} containing elements to be added to this list
     */
    public void addAll(PairList<K, V> other) {
        if (other != null) {
            keys.addAll(other.keys);
            values.addAll(other.values);
        }
    }

    /**
     * Returns the size of the list.
     *
     * @return the size of the list
     */
    public int size() {
        return keys.size();
    }

    /**
     * Checks whether the list is empty.
     *
     * @return whether the list is empty
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Returns the key-value pair at the specified position in this list.
     *
     * @param index the index of the element to return
     * @return the key-value pair at the specified position in this list
     */
    public Pair<K, V> get(int index) {
        return new Pair<>(keys.get(index), values.get(index));
    }

    /**
     * Returns the value for the first key found in the list.
     *
     * @param key the key of the element to get
     * @return the value for the first key found in the list
     */
    public V get(K key) {
        int index = keys.indexOf(key);
        if (index == -1) {
            return null;
        }
        return values.get(index);
    }

    /**
     * Returns the key at the specified position in this list.
     *
     * @param index the index of the element to return
     * @return the key at the specified position in this list
     */
    public K keyAt(int index) {
        return keys.get(index);
    }

    /**
     * Returns the value at the specified position in this list.
     *
     * @param index the index of the element to return
     * @return the value at the specified position in this list
     */
    public V valueAt(int index) {
        return values.get(index);
    }

    /**
     * Returns all keys of the list.
     *
     * @return all keys of the list
     */
    public List<K> keys() {
        return keys;
    }

    /**
     * Returns all values of the list.
     *
     * @return all values of the list
     */
    public List<V> values() {
        return values;
    }

    /**
     * Returns an array containing all of the keys in this list in proper sequence (from first to
     * last element); the runtime type of the returned array is that of the specified array.
     *
     * <p>If the list fits in the specified array, it is returned therein. Otherwise, a new array is
     * allocated with the runtime type of the specified array and the size of this list.
     *
     * @param target the array into which the keys of this list are to be stored, if it is big
     *     enough; otherwise, a new array of the same runtime type is allocated for this purpose.
     * @return an array containing the keys of this list
     */
    public K[] keyArray(K[] target) {
        return keys.toArray(target);
    }

    /**
     * Returns an array containing all of the values in this list in proper sequence (from first to
     * last element); the runtime type of the returned array is that of the specified array.
     *
     * <p>If the list fits in the specified array, it is returned therein. Otherwise, a new array is
     * allocated with the runtime type of the specified array and the size of this list.
     *
     * @param target the array into which the values of this list are to be stored, if it is big
     *     enough; otherwise, a new array of the same runtime type is allocated for this purpose.
     * @return an array containing the values of this list
     */
    public V[] valueArray(V[] target) {
        return values.toArray(target);
    }

    /**
     * Removes the key-value pair for the first key found in the list.
     *
     * @param key the key of the element to be removed
     * @return the value of the removed element, {@code null} if not found
     */
    public V remove(K key) {
        int index = keys.indexOf(key);
        if (index == -1) {
            return null;
        }
        return remove(index);
    }

    /**
     * Removes the key-value pair at an index.
     *
     * @param index the index of the element to remove
     * @return the value of the removed element, {@code null} if not found
     */
    public V remove(int index) {
        keys.remove(index);
        return values.remove(index);
    }

    /**
     * Returns a view of the portion of this PairList between the specified {@code fromIndex}
     * inclusive, and to the end.
     *
     * @param fromIndex the start index (inclusive)
     * @return a view of the portion of this PairList
     */
    public PairList<K, V> subList(int fromIndex) {
        return subList(fromIndex, size());
    }

    /**
     * Returns a view of the portion of this PairList between the specified {@code fromIndex}
     * inclusive, and {@code toIndex}, exclusive.
     *
     * @param fromIndex the start index (inclusive)
     * @param toIndex the end index (exclusive)
     * @return a view of the portion of this PairList
     */
    public PairList<K, V> subList(int fromIndex, int toIndex) {
        List<K> subKeys = keys.subList(fromIndex, toIndex);
        List<V> subValues = values.subList(fromIndex, toIndex);
        return new PairList<>(subKeys, subValues);
    }

    /**
     * Returns the {@link Stream} type of the PairList.
     *
     * @return a {@link Stream} of PairList
     */
    public Stream<Pair<K, V>> stream() {
        return StreamSupport.stream(spliterator(), false);
    }

    /**
     * Returns {@code true} if this list contains the specified key.
     *
     * @param key the key whose presence will be tested
     * @return {@code true} if this list contains the specified key
     */
    public boolean contains(K key) {
        return keys.contains(key);
    }

    /**
     * Removes all duplicate values from the list.
     *
     * @return a new {@code PairList} with the duplicate values removed, taking the latest value for
     *     each key
     */
    public PairList<K, V> unique() {
        return new PairList<>(toMap(false));
    }

    /**
     * Returns a {@code Map} that contains the key-value mappings of this list.
     *
     * @return a {@code Map} that contains the key-value mappings of this list
     */
    public Map<K, V> toMap() {
        return toMap(true);
    }

    /**
     * Returns a {@code Map} that contains the key-value mappings of this list.
     *
     * @param checkDuplicate whether to check for duplicated keys in the list
     * @return a {@code Map} that contains the key-value mappings of this list
     */
    public Map<K, V> toMap(boolean checkDuplicate) {
        int size = keys.size();
        Map<K, V> map = new ConcurrentHashMap<>(size * 3 / 2);
        for (int i = 0; i < size; ++i) {
            if (map.put(keys.get(i), values.get(i)) != null && checkDuplicate) {
                throw new IllegalStateException("Duplicate keys: " + keys.get(i));
            }
        }
        return map;
    }

    @Override
    public Iterator<Pair<K, V>> iterator() {
        return new Itr();
    }

    /** Internal Iterator implementation. */
    private class Itr implements Iterator<Pair<K, V>> {

        private int cursor;
        private int size = size();

        Itr() {}

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return cursor < size;
        }

        /** {@inheritDoc} */
        @Override
        public Pair<K, V> next() {
            if (cursor >= size) {
                throw new NoSuchElementException();
            }

            return get(cursor++);
        }
    }
}
