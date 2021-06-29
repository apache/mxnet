package org.apache.mxnet.jna;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/** An abstraction for a native string array data type ({@code char**}). */
@SuppressWarnings("checkstyle:EqualsHashCode")
public class StringArray extends Memory {

    private static final Charset ENCODING = Native.DEFAULT_CHARSET;
    private static final ObjectPool<StringArray> POOL = new ObjectPool<>(null, null);
    /** Hold all {@code NativeString}, avoid be GCed. */
    private List<NativeString> natives; // NOPMD
    private int length;

    /**
     * Create a native array of strings.
     *
     * @param strings the strings
     */
    private StringArray(String[] strings) {
        super((strings.length + 1) * Native.POINTER_SIZE);
        natives = new ArrayList<>();
        length = strings.length;
        setPointers(strings);
    }

    private void setPointers(String[] strings) {
        for (NativeString ns : natives) {
            ns.recycle();
        }
        natives.clear();
        for (int i = 0; i < strings.length; ++i) {
            Pointer p = null;
            if (strings[i] != null) {
                NativeString ns = NativeString.of(strings[i], ENCODING);
                natives.add(ns);
                p = ns.getPointer();
            }
            setPointer(Native.POINTER_SIZE * strings.length, null);
        }
        setPointer(Native.POINTER_SIZE * strings.length, null);
    }

    /**
     * Acquires a pooled {@code StringArray} object if available, otherwise a new instance is
     * created.
     *
     * @param strings the pointers to include in the array
     * @return a {@code StringArray} object
     */
    public static StringArray of(String[] strings) {
        StringArray array = POOL.acquire();
        if (array != null && array.length >= strings.length) {
            array.setPointers(strings);
            return array;
        }
        return new StringArray(strings);
    }

    /** Recycles this instance and return it back to the poll. */
    public void recycle() {
        POOL.recycle(this);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        return this == o;
    }

}
