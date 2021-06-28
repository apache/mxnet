package org.apache.mxnet.jna;

import com.sun.jna.Function;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * An abstraction for a native pointer array data type ({@code void**}).
 *
 * @see Pointer
 * @see com.sun.jna.ptr.PointerByReference
 * @see Function
 */
@SuppressWarnings("checkstyle:EqualsHashCode")
final class PointerArray extends Memory {

    private static final ObjectPool<PointerArray> POOL = new ObjectPool<>(null, null);

    private int length;

    /**
     * Constructs a {@link Memory} buffer PointerArray given the Pointers to include in it.
     *
     * @param arg the pointers to include in the array
     */
    private PointerArray(Pointer... arg) {
        super(Native.POINTER_SIZE * (arg.length + 1));
        length = arg.length;
        setPointers(arg);
    }

    /**
     * Acquires a pooled {@code PointerArray} object if available, otherwise a new instance is
     * created.
     *
     * @param arg the pointers to include in the array
     * @return a {@code PointerArray} object
     */
    public static PointerArray of(Pointer... arg) {
        PointerArray array = POOL.acquire();
        if (array != null && array.length >= arg.length) {
            array.setPointers(arg);
            return array;
        }
        return new PointerArray(arg);
    }

    /** Recycles this instance and return it back to the pool. */
    public void recycle() {
        POOL.recycle(this);
    }

    private void setPointers(Pointer[] pointers) {
        for (int i = 0; i < pointers.length; i++) {
            setPointer(i * Native.POINTER_SIZE, pointers[i]);
        }
        setPointer(Native.POINTER_SIZE * length, null);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        return o == this;
    }
}