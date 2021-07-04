package org.apache.mxnet.api.ndarray;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.engine.Engine;

import java.nio.ByteBuffer;

// TODO
/**
 * NDArray managers are used to create <I>NDArrays</I> (n-dimensional array on native engine).
 *
 * <p>NDManager is implemented in each deep learning {@link Engine}. {@link NDArray}s are resources
 * that are allocated in each deep learning engine's native memory space. NDManager is the key class
 * that manages these native resources.
 *
 * <p>NDArray can only be created through NDManager. By default, NDArray's lifecycle is attached to
 * the creator NDManager. NDManager itself implements {@link AutoCloseable}. When NDManager is
 * closed, all the resource associated with it will be closed as well.
 *
 * <p>A typical place to obtain NDManager is in {@link Translator#processInput(TranslatorContext,
 * Object)} or {@link Translator#processOutput(TranslatorContext, NDList)}.
 *
 * <p>The following is an example of how to use NDManager:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;FloatBuffer, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, FloatBuffer input) {
 *         <b>NDManager manager = ctx.getNDManager();</b>
 *         NDArray array = <b>manager</b>.create(shape);
 *         array.set(input);
 *         return new NDList(array);
 *     } // NDArrays created in this method will be closed after method return.
 * }
 * </pre>
 *
 * <p>NDManager has a hierarchical structure; it has a single parent NDManager and has child
 * NDManagers. When the parent NDManager is closed, all children will be closed as well.
 *
 * <p>The DJL engine manages NDManager's lifecycle by default. You only need to manage the user
 * created child NDManager. The child NDManager becomes useful when you create a large number of
 * temporary NDArrays and want to free the resources earlier than the parent NDManager's lifecycle.
 *
 * <p>The following is an example of such a use case:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;List&lt;FloatBuffer&gt;&gt;, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, List&lt;FloatBuffer&gt; input) {
 *         NDManager manager = ctx.getNDManager();
 *         NDArray array = manager.create(shape, dataType);
 *         for (int i = 0; i &lt; input.size(); ++i) {
 *             try (<b>NDManager childManager = manager.newSubManager()</b>) {
 *                  NDArray tmp = <b>childManager</b>.create(itemShape);
 *                  tmp.put(input.get(i);
 *                  array.put(i, tmp);
 *             } // NDArray <i>tmp</i> will be closed here
 *         }
 *         return new NDList(array);
 *     }
 * }
 * </pre>
 *
 * <p>You can also close an individual NDArray. NDManager won't close an NDArray that's already been
 * closed. In certain use cases, you might want to return an NDArray outside of NDManager's scope.
 *
 * @see NDArray
 * @see Translator
 * @see TranslatorContext#getNDManager()
 * @see <a
 *     href="https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md">NDArray
 *     Memory Management Guide</a>
 */
public interface NDManager extends AutoCloseable {

    /**
     * Creates a new top-level {@code NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    static NDManager newBaseManager() {
        return Engine.getInstance().newBaseManager();
    }

    /**
     * Creates a new top-level {@code NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    static NDManager newBaseManager(Device device) {
        return Engine.getInstance().newBaseManager(device);
    }

    /**
     * Creates a new manager based on the given resource.
     *
     * @param resource the resource to use
     * @return a new memory scrope containing the array
     */
    static NDManager from(NDResource resource) {
        return resource.getManager().newSubManager();
    }

    /**
     * Allocates a new engine specific direct byte buffer.
     *
     * @param capacity the new buffer's capacity, in bytes
     * @return the new byte buffer
     */
    ByteBuffer allocateDirect(int capacity);

    /**
     * Creates a child {@code NDManager}.
     *
     * <p>Child {@code NDManager} will inherit default {@link Device} from this {@code NDManager}.
     *
     * @return a child {@code NDManager}
     */
    NDManager newSubManager();

    /**
     * Creates a child {@code NDManager} with specified default {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a child {@code NDManager}
     */
    NDManager newSubManager(Device device);

    /** {@inheritDoc} */
    @Override
    void close();

}
