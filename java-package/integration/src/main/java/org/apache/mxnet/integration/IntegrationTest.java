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

package org.apache.mxnet.integration;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.integration.util.Arguments;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class IntegrationTest {

    private static final Logger logger = LoggerFactory.getLogger(IntegrationTest.class);

    private Class<?> source;

    public IntegrationTest(Class<?> source) {
        this.source = source;
    }

    public static void main(String[] args) {
        new IntegrationTest(IntegrationTest.class).runTests(args);
        // TODO: not elegant solution to native library crash
        //        System.exit(0);
    }

    public boolean runTests(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            Duration duration = Duration.ofMinutes(arguments.getDuration());
            List<TestClass> tests = listTests(arguments, source);

            boolean testsPassed = true;
            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();

                testsPassed = testsPassed && runTests(tests);

                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
            }
            return testsPassed;
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
            return false;
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
            return false;
        }
    }

    private boolean runTests(List<TestClass> tests) {
        Map<TestResult, Integer> totals = new ConcurrentHashMap<>();
        for (TestClass testClass : tests) {
            logger.info("Running test {} ...", testClass.getName());
            int testCount = testClass.getTestCount();

            try {
                if (!testClass.beforeClass()) {
                    totals.merge(TestResult.FAILED, testCount, Integer::sum);
                    continue;
                }

                for (int i = 0; i < testCount; ++i) {
                    TestResult result = testClass.runTest(i);
                    totals.merge(result, 1, Integer::sum);
                }
            } finally {
                testClass.afterClass();
            }
        }

        int totalFailed = totals.getOrDefault(TestResult.FAILED, 0);
        int totalPassed = totals.getOrDefault(TestResult.SUCCESS, 0);
        int totalSkipped = totals.getOrDefault(TestResult.SKIPPED, 0);
        int totalUnsupported = totals.getOrDefault(TestResult.UNSUPPORTED, 0);
        if (totalSkipped > 0) {
            logger.info("Skipped: {} tests", totalSkipped);
        }
        if (totalUnsupported > 0) {
            logger.info("Unsupported: {} tests", totalUnsupported);
        }
        if (totalFailed > 0) {
            logger.error("Failed {} out of {} tests", totalFailed, totalFailed + totalPassed);
        } else {
            logger.info("Passed all {} tests", totalPassed);
        }
        return totalFailed == 0;
    }

    private static List<TestClass> listTests(Arguments arguments, Class<?> source)
            throws IOException, ReflectiveOperationException, URISyntaxException {
        String className = arguments.getClassName();
        String methodName = arguments.getMethodName();
        List<TestClass> tests = new ArrayList<>();
        try {
            if (className != null) {
                Class<?> clazz;
                if (className.startsWith(arguments.getPackageName())) {
                    clazz = Class.forName(className);
                } else {
                    clazz = Class.forName(arguments.getPackageName() + className);
                }
                getTestsInClass(clazz, methodName).map(tests::add);
            } else {
                List<Class<?>> classes = listTestClasses(arguments, source);
                for (Class<?> clazz : classes) {
                    getTestsInClass(clazz, methodName).map(tests::add);
                }
            }
        } catch (ReflectiveOperationException | IOException | URISyntaxException e) {
            logger.error("Failed to resolve test class.", e);
            throw e;
        }
        return tests;
    }

    private static Optional<TestClass> getTestsInClass(Class<?> clazz, String methodName)
            throws ReflectiveOperationException {
        if (clazz.getConstructors().length == 0) {
            return Optional.empty();
        }
        Constructor<?> ctor = clazz.getConstructor();
        Object obj = ctor.newInstance();
        TestClass testClass = new TestClass(obj);

        for (Method method : clazz.getDeclaredMethods()) {
            Test testMethod = method.getAnnotation(Test.class);
            if (testMethod != null) {
                if (testMethod.enabled()
                        && (methodName == null || methodName.equals(method.getName()))) {
                    testClass.addTestMethod(method);
                }
                continue;
            }
            BeforeClass beforeClass = method.getAnnotation(BeforeClass.class);
            if (beforeClass != null) {
                testClass.addBeforeClass(method);
                continue;
            }
            AfterClass afterClass = method.getAnnotation(AfterClass.class);
            if (afterClass != null) {
                testClass.addAfterClass(method);
                continue;
            }
            BeforeTest beforeTest = method.getAnnotation(BeforeTest.class);
            if (beforeTest != null) {
                testClass.addBeforeTest(method);
                continue;
            }
            AfterTest afterTest = method.getAnnotation(AfterTest.class);
            if (afterTest != null) {
                testClass.addAfterTest(method);
            }
        }

        return Optional.of(testClass);
    }

    private static List<Class<?>> listTestClasses(Arguments arguments, Class<?> clazz)
            throws IOException, ClassNotFoundException, URISyntaxException {
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
                if (className.startsWith(arguments.getPackageName()) && !className.contains("$")) {
                    try {
                        classList.add(Class.forName(className));
                    } catch (ExceptionInInitializerError ignore) {
                        // ignore
                    }
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
                        if (fileName.startsWith(arguments.getPackageName())) {
                            try {
                                classList.add(Class.forName(fileName));
                            } catch (ExceptionInInitializerError ignore) {
                                // ignore
                            }
                        }
                    }
                }
            }
        }

        return classList;
    }

    private static final class TestClass {

        private Object object;
        private List<Method> testMethods;
        private List<Method> beforeClass;
        private List<Method> afterClass;
        private List<Method> beforeTest;
        private List<Method> afterTest;

        public TestClass(Object object) {
            this.object = object;
            testMethods = new ArrayList<>();
            beforeClass = new ArrayList<>();
            afterClass = new ArrayList<>();
            beforeTest = new ArrayList<>();
            afterTest = new ArrayList<>();
        }

        public void addTestMethod(Method method) {
            testMethods.add(method);
        }

        public void addBeforeClass(Method method) {
            beforeClass.add(method);
        }

        public void addAfterClass(Method method) {
            afterClass.add(method);
        }

        public void addBeforeTest(Method method) {
            beforeTest.add(method);
        }

        public void addAfterTest(Method method) {
            afterTest.add(method);
        }

        public boolean beforeClass() {
            try {
                for (Method method : beforeClass) {
                    method.invoke(object);
                }
                return true;
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
            return false;
        }

        public void afterClass() {
            try {
                for (Method method : afterClass) {
                    method.invoke(object);
                }
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
        }

        public boolean beforeTest() {
            try {
                for (Method method : beforeTest) {
                    method.invoke(object);
                }
                return true;
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
            return false;
        }

        public void afterTest() {
            try {
                for (Method method : afterTest) {
                    method.invoke(object);
                }
            } catch (InvocationTargetException | IllegalAccessException e) {
                logger.error("", e.getCause());
            }
        }

        public TestResult runTest(int index) {
            if (!beforeTest()) {
                return TestResult.FAILED;
            }

            TestResult result;
            Method method = testMethods.get(index);
            try {
                long begin = System.nanoTime();
                method.invoke(object);
                String time = String.format("%.3f", (System.nanoTime() - begin) / 1000_0000f);
                logger.info("Test {}.{} PASSED, duration: {}", getName(), method.getName(), time);
                result = TestResult.SUCCESS;
            } catch (IllegalAccessException | InvocationTargetException e) {
                if (expectedException(method, e)) {
                    logger.info("Test {}.{} PASSED", getName(), method.getName());
                    result = TestResult.SUCCESS;
                } else if (e.getCause() instanceof SkipException) {
                    logger.info("Test {}.{} SKIPPED", getName(), method.getName());
                    result = TestResult.SKIPPED;
                } else if (e.getCause() instanceof UnsupportedOperationException) {
                    logger.info("Test {}.{} UNSUPPORTED", getName(), method.getName());
                    logger.trace("", e.getCause());
                    result = TestResult.UNSUPPORTED;
                } else {
                    logger.error("Test {}.{} FAILED", getName(), method.getName());
                    logger.error("", e.getCause());
                    result = TestResult.FAILED;
                }
            } finally {
                afterTest();
            }
            return result;
        }

        public int getTestCount() {
            return testMethods.size();
        }

        public String getName() {
            return object.getClass().getName();
        }

        private static boolean expectedException(Method method, Exception e) {
            Test test = method.getAnnotation(Test.class);
            Class<?>[] exceptions = test.expectedExceptions();
            if (exceptions.length > 0) {
                Throwable exception = e.getCause();
                for (Class<?> c : exceptions) {
                    if (c.isInstance(exception)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    public enum TestResult {
        SUCCESS,
        FAILED,
        SKIPPED,
        UNSUPPORTED;
    }
}
