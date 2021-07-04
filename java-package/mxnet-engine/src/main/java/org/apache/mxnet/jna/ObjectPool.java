package org.apache.mxnet.jna;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class ObjectPool<T> {

    private Queue<T> queue;
    private Supplier<T> supplier;
    private Consumer<T> consumer;

    public ObjectPool(Supplier<T> supplier, Consumer<T> consumer) {
        queue = new ConcurrentLinkedQueue<>();
        this.supplier = supplier;
        this.consumer = consumer;
    }

    public T acquire() {
        T item = queue.poll();
        if (item == null) {
            if (supplier != null) {
                return supplier.get();
            }
        }
        return item;
    }

    public void recycle(T item) {
        if (consumer != null) {
            consumer.accept(item);
        }
        queue.add(item);
    }
}
