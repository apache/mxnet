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
package org.apache.mxnet.integration.util;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;

public final class CoverageUtils {

    private CoverageUtils() {}

    public static void testGetterSetters(Class<?> baseClass)
            throws IOException, ReflectiveOperationException, URISyntaxException {
        List<Class<?>> list = getClasses(baseClass);
        for (Class<?> clazz : list) {
            Object obj = null;
            if (clazz.isEnum()) {
                obj = clazz.getEnumConstants()[0];
            } else {
                Constructor<?>[] constructors = clazz.getConstructors();
                for (Constructor<?> con : constructors) {
                    try {
                        Class<?>[] types = con.getParameterTypes();
                        Object[] args = new Object[types.length];
                        for (int i = 0; i < args.length; ++i) {
                            args[i] = getMockInstance(types[i], true);
                        }
                        con.setAccessible(true);
                        obj = con.newInstance(args);
                    } catch (ReflectiveOperationException ignore) {
                        // ignore
                    }
                }
            }
            if (obj == null) {
                continue;
            }

            Method[] methods = clazz.getDeclaredMethods();
            for (Method method : methods) {
                String methodName = method.getName();
                int parameterCount = method.getParameterCount();
                try {
                    if (parameterCount == 0
                            && (methodName.startsWith("get")
                                    || methodName.startsWith("is")
                                    || "toString".equals(methodName)
                                    || "hashCode".equals(methodName))) {
                        method.invoke(obj);
                    } else if (parameterCount == 1
                            && (methodName.startsWith("set") || "fromValue".equals(methodName))) {
                        Class<?> type = method.getParameterTypes()[0];
                        method.invoke(obj, getMockInstance(type, true));
                    } else if ("equals".equals(methodName)) {
                        method.invoke(obj, obj);
                        method.invoke(obj, (Object) null);
                        Class<?> type = method.getParameterTypes()[0];
                        method.invoke(obj, getMockInstance(type, true));
                    }
                } catch (ReflectiveOperationException ignore) {
                    // ignore
                }
            }
        }
    }

    private static List<Class<?>> getClasses(Class<?> clazz)
            throws IOException, ReflectiveOperationException, URISyntaxException {
        ClassLoader appClassLoader = Thread.currentThread().getContextClassLoader();
        Field field = appClassLoader.getClass().getDeclaredField("ucp");
        field.setAccessible(true);
        Object ucp = field.get(appClassLoader);
        Method method = ucp.getClass().getDeclaredMethod("getURLs");
        URL[] urls = (URL[]) method.invoke(ucp);
        ClassLoader cl = new TestClassLoader(urls, Thread.currentThread().getContextClassLoader());

        URL url = clazz.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();

        if (!"file".equalsIgnoreCase(url.getProtocol())) {
            return Collections.emptyList();
        }

        List<Class<?>> classList = new ArrayList<>();

        Path classPath = Paths.get(url.toURI());
        if (Files.isDirectory(classPath)) {
            Collection<Path> files =
                    Files.walk(classPath)
                            .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                            .collect(Collectors.toList());
            for (Path file : files) {
                Path p = classPath.relativize(file);
                String className = p.toString();
                className = className.substring(0, className.lastIndexOf('.'));
                className = className.replace(File.separatorChar, '.');

                try {
                    classList.add(Class.forName(className, true, cl));
                } catch (Error ignore) {
                    // ignore
                }
            }
        } else if (path.toLowerCase().endsWith(".jar")) {
            try (JarFile jarFile = new JarFile(classPath.toFile())) {
                Enumeration<JarEntry> en = jarFile.entries();
                while (en.hasMoreElements()) {
                    JarEntry entry = en.nextElement();
                    String fileName = entry.getName();
                    if (fileName.endsWith(".class")) {
                        fileName = fileName.substring(0, fileName.lastIndexOf('.'));
                        fileName = fileName.replace('/', '.');
                        try {
                            classList.add(Class.forName(fileName, true, cl));
                        } catch (Error ignore) {
                            // ignore
                        }
                    }
                }
            }
        }

        return classList;
    }

    private static Object getMockInstance(Class<?> clazz, boolean useConstructor) {
        if (clazz.isPrimitive()) {
            if (clazz == Boolean.TYPE) {
                return Boolean.TRUE;
            }
            if (clazz == Character.TYPE) {
                return '0';
            }
            if (clazz == Byte.TYPE) {
                return (byte) 0;
            }
            if (clazz == Short.TYPE) {
                return (short) 0;
            }
            if (clazz == Integer.TYPE) {
                return 0;
            }
            if (clazz == Long.TYPE) {
                return 0L;
            }
            if (clazz == Float.TYPE) {
                return 0f;
            }
            if (clazz == Double.TYPE) {
                return 0d;
            }
        }

        if (clazz.isAssignableFrom(String.class)) {
            return "";
        }

        if (clazz.isAssignableFrom(List.class)) {
            return new ArrayList<>();
        }

        if (clazz.isAssignableFrom(Set.class)) {
            return new HashSet<>();
        }

        if (clazz.isAssignableFrom(Map.class)) {
            return new HashMap<>();
        }

        if (clazz.isEnum()) {
            return clazz.getEnumConstants()[0];
        }

        if (clazz.isInterface()) {
            return newProxyInstance(clazz);
        }

        if (useConstructor) {
            Constructor<?>[] constructors = clazz.getConstructors();
            for (Constructor<?> con : constructors) {
                try {
                    Class<?>[] types = con.getParameterTypes();
                    Object[] args = new Object[types.length];
                    for (int i = 0; i < args.length; ++i) {
                        args[i] = getMockInstance(types[i], false);
                    }
                    con.setAccessible(true);
                    return con.newInstance(args);
                } catch (ReflectiveOperationException ignore) {
                    // ignore
                }
            }
        }

        return null;
    }

    @SuppressWarnings({"rawtypes", "PMD.UseProperClassLoader"})
    private static Object newProxyInstance(Class<?> clazz) {
        ClassLoader cl = clazz.getClassLoader();
        return Proxy.newProxyInstance(cl, new Class[] {clazz}, (proxy, method, args) -> null);
    }

    private static final class TestClassLoader extends URLClassLoader {

        public TestClassLoader(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        /** {@inheritDoc} */
        @Override
        public Class<?> loadClass(String name) throws ClassNotFoundException {
            try {
                return findClass(name);
            } catch (ClassNotFoundException e) {
                ClassLoader classLoader = getParent();
                if (classLoader == null) {
                    classLoader = getSystemClassLoader();
                }
                return classLoader.loadClass(name);
            }
        }
    }
}
