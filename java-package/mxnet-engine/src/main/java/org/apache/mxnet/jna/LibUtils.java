/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet.jna;

import com.sun.jna.Library;
import com.sun.jna.Native;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.util.Platform;
import org.apache.mxnet.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
@SuppressWarnings("MissingJavadocMethod")
public final class LibUtils {

    private static final Logger logger = LoggerFactory.getLogger(LibUtils.class);

    private static final String LIB_NAME = "mxnet";

    private static final String MXNET_LIBRARY_PATH = "MXNET_LIBRARY_PATH";

    private static final String MXNET_PROPERTIES_FILE_PATH = "native/lib/mxnet.properties";

    private LibUtils() {}

    public static MxnetLibrary loadLibrary() {

        String libName = getLibName();
        logger.debug("Loading mxnet library from: {}", libName);
        if (System.getProperty("os.name").startsWith("Linux")) {
            logger.info("Loading on Linux platform");
            Map<String, Integer> options = new ConcurrentHashMap<>();
            int rtld = 1; // Linux RTLD lazy + local
            options.put(Library.OPTION_OPEN_FLAGS, rtld);
            return Native.load(libName, MxnetLibrary.class, options);
        }
        return Native.load(libName, MxnetLibrary.class);
    }

    public static String getLibName() {
        String libName = findOverrideLibrary();
        if (libName == null) {
            libName = LibUtils.findLibraryInClasspath();
            if (libName == null) {
                libName = LIB_NAME;
            }
        }

        return libName;
    }

    private static String findOverrideLibrary() {
        // TODO: load from jar files
        String libPath = System.getenv(MXNET_LIBRARY_PATH);
        if (libPath != null) {
            String libName = findLibraryInPath(libPath);
            if (libName != null) {
                return libName;
            }
        }

        libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            return findLibraryInPath(libPath);
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

        // Find the mxnet library version that matches local system platform
        // throw exception if no one matches
        Platform systemPlatform = Platform.fromSystem();
        try {
            while (urls.hasMoreElements()) {
                URL url = urls.nextElement();
                Platform platform = Platform.fromUrl(url);
                if (!platform.isPlaceholder() && platform.matches(systemPlatform)) {
                    return loadLibraryFromClasspath(platform);
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
            return Thread.currentThread()
                    .getContextClassLoader()
                    .getResources(MXNET_PROPERTIES_FILE_PATH);
        } catch (IOException e) {
            logger.warn(
                    String.format(
                            "IO Exception occurs when try to find the file %s", MXNET_LIBRARY_PATH),
                    e);
            return null;
        }
    }

    private static String loadLibraryFromClasspath(Platform platform) {
        Path tmp = null;
        try {
            String libName = System.mapLibraryName(LIB_NAME);
            Path cacheFolder = Utils.getEngineCacheDir(LIB_NAME);
            logger.debug("Using cache dir: {}", cacheFolder);

            Path dir = cacheFolder.resolve(platform.getVersion() + platform.getClassifier());
            Path path = dir.resolve(libName);
            if (Files.exists(path)) {
                return path.toAbsolutePath().toString();
            }
            Files.createDirectories(cacheFolder);
            tmp = Files.createTempDirectory(cacheFolder, "tmp");
            for (String file : platform.getLibraries()) {
                String libPath = "/native/lib/" + file;
                try (InputStream is = LibUtils.class.getResourceAsStream(libPath)) {
                    logger.info("Extracting {} to cache ...", file);
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Utils.moveQuietly(tmp, dir);
            return path.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract MXNet native library", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    private static String findLibraryInPath(String libPath) {
        String[] paths = libPath.split(File.pathSeparator);
        List<String> mappedLibNames;
        if (com.sun.jna.Platform.isMac()) {
            mappedLibNames = Arrays.asList("libmxnet.dylib", "libmxnet.jnilib", "libmxnet.so");
        } else {
            mappedLibNames = Collections.singletonList(System.mapLibraryName(LIB_NAME));
        }

        for (String path : paths) {
            File p = new File(path);
            if (!p.exists()) {
                continue;
            }
            for (String name : mappedLibNames) {
                if (p.isFile() && p.getName().endsWith(name)) {
                    return p.getAbsolutePath();
                }

                File file = new File(path, name);
                if (file.exists() && file.isFile()) {
                    return file.getAbsolutePath();
                }
            }
        }
        return null;
    }
}
