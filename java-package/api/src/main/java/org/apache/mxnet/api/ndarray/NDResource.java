package org.apache.mxnet.api.ndarray;

public interface NDResource extends AutoCloseable {

    /**
     * Returns the {@link NDManager} that manages this.
     *
     * @return the {@link NDManager} that manages this.
     */
    NDManager getManager();

    /**
     * Attaches this {@link NDResource} to the specified {@link NDManager}.
     *
     * <p>Attached resource will be closed when the {@link NDManager} is closed.
     *
     * @param manager the {@link NDManager} to be attached to
     */
    void attach(NDManager manager);

    /**
     * Temporarily attaches this {@link NDResource} to the specified {@link NDManager}.
     *
     * <p>Attached resource will be returned to the original manager when the {@link NDManager} is
     * closed.
     *
     * @param manager the {@link NDManager} to be attached to
     */
    void tempAttach(NDManager manager);

    /**
     * Detaches the {@link NDResource} from current {@link NDManager}'s lifecycle.
     *
     * <p>This becomes un-managed and it is the user's responsibility to close this. Failure to
     * close the resource might cause your machine to run out of native memory.
     *
     * @see NDManager
     */
    void detach();

    /** {@inheritDoc} */
    @Override
    void close();

}
