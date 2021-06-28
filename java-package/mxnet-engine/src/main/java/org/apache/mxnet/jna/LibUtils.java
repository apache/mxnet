package org.apache.mxnet.jna;

import org.apache.mxnet.util.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Enumeration;
import java.util.regex.Pattern;
import java.net.URL;

/**
 * Utilities for finding the MXNet Engine binary on the System.
 *
 * <p>The Engine will be searched for in a variety of locations in the following order:
 *
 * <ol>
 *   <li>In the path specified by the MXNET_LIBRARY_PATH environment variable
 *   <li>In a jar file location in the classpath. These jars can be created with the mxnet-native
 *       module.
 *   <li>In the python3 path. These can be installed using pip.
 *   <li>In the python path. These can be installed using pip.
 * </ol>
 */

public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "mxnet";

    private static final String MXNET_LIBRARY_PATH = "MXNET_LIBRARY_PATH";

    private static final String MXNET_PROPERTIES_FILE_PATH = "native/lib/mxnet.properties";

    private static final Pattern VERSION_PATTERN =
            Pattern.compile("(\\d+\\.\\d+\\.\\d+(-[a-z]+)?)(-SNAPSHOT)?(-\\d+)?");

    private LibUtils() {}

    public static MxnetLibrary loadLibrary() {
        // TODO
        String libName = getLibName();
        return null;
    }

    public static String getLibName() {
        // TODO
        return null;
    }

    private static String findOverrideLibrary() {
        String libPath = System.getenv(MXNET_LIBRARY_PATH);
        if (libPath != null) {
            // TODO
        }

        return null;
    }

    private static synchronized String findLibraryInClasspath() {
        Enumeration<URL> urls = getUrls();
        // No native jars
        if (!urls.hasMoreElements()) {
            logger.debug("mxnet.properties not found in class path.");
            return null;
        }

        Platform systemPlatform = Platform.fromSystem();
        try {
            Platform matching = null;
            Platform placeholder = null;
            while (urls.hasMoreElements()) {
                URL url = urls.nextElement();
                Platform platform = Platform.fromUrl(url);
                if (platform.isPlaceholder()) {
                    placeholder = platform;
                } else if (platform.matches(systemPlatform)) {
                    matching = platform;
                    break;
                }
            }

            if (matching != null) {
                return loadLibraryFromClasspath(matching);
            }

            if (placeholder != null) {
                try {
                    return downloadMxnet(placeholder);
                } catch (IOException e) {
                    throw new IllegalStateException("Failed to download MXNet native library", e);
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read MXNet native library jar properties", e);
        }

        throw new IllegalStateException(
                "Your MXNet native library jar does not match your operating system. Make sure that the Maven Dependency Classifier matches your system type.");
    }

    private static Enumeration<URL> getUrls() {
        try {
            Enumeration<URL> urls =
                    Thread.currentThread()
                            .getContextClassLoader()
                            .getResources(MXNET_PROPERTIES_FILE_PATH);
            return urls;
        } catch (IOException e) {
            logger.warn("IO Exception occurs when try to find the file %s", MXNET_LIBRARY_PATH, e);
            return null;
        }
    }

    // TODO
    private static String loadLibraryFromClasspath(Platform platform) {
        return null;
    }

    //TODO
    private static String findLibraryInPath(String libPath) {
        return null;
    }

    //TODO
    private static String downloadMxnet(Platform platform) throws IOException{
        return null;
    }

    //TODO
    private static boolean notSupported(Platform platform) {
        // to be loaded from properties
        return false;
    }



}
