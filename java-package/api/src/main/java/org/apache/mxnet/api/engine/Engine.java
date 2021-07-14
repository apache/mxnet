package org.apache.mxnet.api.engine;

import org.apache.mxnet.api.util.cuda.CudaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.ServiceLoader;


//TODO
public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static EngineProvider ENGINE_PROVIDE = initEngineProvider();

    private Device defaultDevice;

    // use object to check if it's set
    private Integer seed;

    private static synchronized EngineProvider initEngineProvider() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        Iterator<EngineProvider> loaderIterator = loaders.iterator();
        if (loaderIterator.hasNext()) {
            EngineProvider engineProvider = loaderIterator.next();
            logger.debug("Found EngineProvider for engine: {}", engineProvider.getEngineName());
            return engineProvider;
        } else {
            logger.debug("No EngineProvider found");
            return null;
        }
    }

    /**
     * Returns the name of the Engine.
     *
     * @return the name of the engine
     */
    public abstract String getEngineName();

    public static Engine getInstance() {
        return getEngine();
    }

    /**
     * Returns the {@code Engine} with the given name.
     *
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getEngine() {
        if (ENGINE_PROVIDE == null) {
            throw new IllegalArgumentException("Deep learning engine not found");
        }
        return ENGINE_PROVIDE.getEngine();
    }

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return {@code true} if the engine has the specified capability
     */
    public abstract boolean hasCapability(String capability);

    /**
     * Returns the engine's default {@link Device}.
     *
     * @return the engine's default {@link Device}
     */
    public Device defaultDevice() {
        if (defaultDevice == null) {
            if (hasCapability(StandardCapabilities.CUDA) && CudaUtils.getGpuCount() > 0) {
                defaultDevice = Device.gpu();
            } else {
                defaultDevice = Device.cpu();
            }
        }
        return defaultDevice;
    }

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Device device);
}
