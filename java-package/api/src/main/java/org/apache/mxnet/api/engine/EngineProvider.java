package org.apache.mxnet.api.engine;

/**
 * The {@code EngineProvider} instance manufactures an {@link Engine} instance, which is available
 * in the system.
 *
 * <p>At initialization time, the {@link java.util.ServiceLoader} will search for {@code
 * EngineProvider} implementations available in the class path.
 *
 * <p>{@link Engine} is designed as a collection of singletons. {@link Engine#getInstance()} will
 * return the default Engine, which is the first one found in the classpath. Many of the standard
 * APIs will rely on this default Engine instance such as when creating a {@link
 * NDManager} or {@link ai.djl.Model}. However, you can directly get a specific
 * Engine instance (e.g. {@code MxEngine}) by calling {@link Engine#getEngine(String)}.
 */
public interface EngineProvider {

    /**
     * Returns the name of the {@link Engine}.
     *
     * @return the name of {@link Engine}
     */
    String getEngineName();

    /**
     * Returns the rank of the {@link Engine}.
     *
     * @return the rank of {@link Engine}
     */
    int getEngineRank();

    /**
     * Returns the instance of the {@link Engine} class EngineProvider should bind to.
     *
     * @return the instance of {@link Engine}
     */
    Engine getEngine();

}
