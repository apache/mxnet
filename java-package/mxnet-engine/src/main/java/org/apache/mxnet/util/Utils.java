package org.apache.mxnet.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** A class containing utility methods. */
public final class Utils {

    private static final int BUFF_SIZE = 81920;
    private static final String ENGINE_CACHE_DIR = "ENGINE_CACHE_DIR";
    private static final String MXNET_CACHE_DIR = "MXNET_CACHE_DIR";

    private Utils() {}

    /**
     * Returns the index of the first occurrence of the specified element in {@code array}, or -1 if
     * this list does not contain the element.
     *
     * @param array the input array
     * @param value the element to search for
     * @param <T> the array type
     * @return the index of the first occurrence of the specified element in {@code array}, or -1 if
     *     this list does not contain the element
     */
    public static <T> int indexOf(T[] array, T value) {
        if (array != null) {
            if (value == null) {
                for (int i = 0; i < array.length; ++i) {
                    if (array[i] == null) {
                        return i;
                    }
                }
            } else {
                for (int i = 0; i < array.length; ++i) {
                    if (value.equals(array[i])) {
                        return i;
                    }
                }
            }
        }
        return -1;
    }

    /**
     * Returns {@code true} if the {@code array} contains the specified element.
     *
     * @param array the input array
     * @param value the element whose presence in {@code array} is to be tested
     * @param <T> the array type
     * @return {@code true} if this list contains the specified element
     */
    public static <T> boolean contains(T[] array, T value) {
        return indexOf(array, value) >= 0;
    }

    /**
     * Adds padding chars to specified StringBuilder.
     *
     * @param sb the StringBuilder to append
     * @param c the padding char
     * @param count the number characters to be added
     */
    public static void pad(StringBuilder sb, char c, int count) {
        for (int i = 0; i < count; ++i) {
            sb.append(c);
        }
    }

    /**
     * Deletes an entire directory and ignore all errors.
     *
     * @param dir the directory to be removed
     */
    public static void deleteQuietly(Path dir) {
        try {
            Files.walk(dir)
                    .sorted(Comparator.reverseOrder())
                    .forEach(
                            path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException ignore) {
                                    // ignore
                                }
                            }
                            );
        } catch (IOException ignore) {
            // ignore
        }
    }

    /**
     * Renames a file to a target file and ignore error if target already exists.
     *
     * @param source the path to the file to move
     * @param target the path to the target file
     * @throws IOException if move file failed
     */
    public static void moveQuietly(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            if (!Files.exists(target)) {
                throw e;
            }
        }
    }

    /**
     * Reads {@code is} as UTF-8 string.
     *
     * @param is the InputStream to be read
     * @return a UTF-8 encoded string
     * @throws IOException if IO error occurs
     */
    public static String toString(InputStream is) throws IOException {
        return null;
    }

    /**
     * Reads {@code is} as byte array.
     *
     * @param is the InputStream to be read
     * @return a byte array
     * @throws IOException if IO error occurs
     */
    public static byte[] toByteArray(InputStream is) throws IOException {

        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(BUFF_SIZE)) {
            byte[] buf = new byte[BUFF_SIZE];
            int read;
            while ((read = is.read(buf)) != -1) {
                bos.write(buf, 0, read);
            }
            return bos.toByteArray();
        }
    }

    /**
     * Reads all lines from a file.
     *
     * @param file the file to be read
     * @return all lines in the file
     * @throws IOException if read file failed
     */
    public static List<String> readLines(Path file) throws IOException {
        return readLines(file, false);
    }

    /**
     * Reads all lines from a file.
     *
     * @param file the file to be read
     * @param trim true if you want to trim the line and exclude empty lines
     * @return all lines in the file
     * @throws IOException if read file failed
     */
    public static List<String> readLines(Path file, boolean trim) throws IOException {
        if (Files.notExists(file)) {
            return Collections.emptyList();
        }
        try (InputStream is = Files.newInputStream(file)) {
            return readLines(is, trim);
        }
    }

    /**
     * Reads all lines from the specified InputStream.
     *
     * @param is the InputStream to read
     * @return all lines from the input
     */
    public static List<String> readLines(InputStream is) {
        return readLines(is, false);
    }

    /**
     * Reads all lines from the specified InputStream.
     *
     * @param is the InputStream to read
     * @param trim true if you want to trim the line and exclude empty lines
     * @return all lines from the input
     */
    public static List<String> readLines(InputStream is, boolean trim) {
        List<String> list = new ArrayList<>();
        try (Scanner scanner =
                new Scanner(is, StandardCharsets.UTF_8.name()).useDelimiter("\\n|\\r\\n")) {
            while (scanner.hasNext()) {
                String line = scanner.next();
                if (trim) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;
                    }
                }
                list.add(line);
            }
        }
        return list;
    }

    /**
     * Converts a List of Number to float array.
     *
     * @param list the list to be converted
     * @return a float array
     */
    public static float[] toFloatArray(List<? extends Number> list) {
        float[] ret = new float[list.size()];
        int idx = 0;
        for (Number n : list) {
            ret[idx++] = n.floatValue();
        }
        return ret;
    }

    /**
     * Gets the current epoch number.
     *
     * @param modelDir the path to the directory where the model files are stored
     * @param modelName the name of the model
     * @return the current epoch number
     * @throws IOException if an I/O error occurs
     */
    public static int getCurrentEpoch(Path modelDir, String modelName) throws IOException {
        final Pattern pattern = Pattern.compile(Pattern.quote(modelName) + "-(\\d{4}).params");
        List<Integer> checkpoints =
                Files.walk(modelDir, 1)
                        .map(
                                p -> {
                                    Matcher m = pattern.matcher(p.toFile().getName());
                                    if (m.matches()) {
                                        return Integer.parseInt(m.group(1));
                                    }
                                    return null;
                                })
                        .filter(Objects::nonNull)
                        .sorted()
                        .collect(Collectors.toList());
        if (checkpoints.isEmpty()) {
            return -1;
        }
        return checkpoints.get(checkpoints.size() - 1);
    }

    /**
     * Utility function to help debug nan values in parameters and their gradients.
     *
     * @param parameters the list of parameters to check
     * @param checkGradient whether to check parameter value or its gradient value
     * @param logger the logger to log the result
     */
    // TODO
//    public static void checkParameterValues(
//            Pairlist<String, Parameter> parameters, boolean checkGradient, Logger logger) {
//
//    }

    /**
     * Utility function to help summarize the values in an {@link NDArray}.
     *
     * @param array the {@link NDArray} to be summarized
     * @param logger the logger to log the result
     * @param prefix the prefix or name to be displayed
     */
    // TODO
//    public static void checkNDArrayValues(NDArray array, Logger logger, String prefix) {
//
//    }

    /**
     * Utility function to get Engine specific cache directory.
     *
     * @param engine the engine name
     * @return DJL engine cache directory
     */
    public static Path getEngineCacheDir(String engine) {
        return getEngineCacheDir().resolve(engine);
    }

    /**
     * Utility function to get Engine cache directory.
     *
     * @return DJL engine cache directory
     */
    public static Path getEngineCacheDir() {
        String cacheDir = System.getProperty(ENGINE_CACHE_DIR);
        if (cacheDir == null || cacheDir.isEmpty()) {
            cacheDir = System.getenv(ENGINE_CACHE_DIR);
            if (cacheDir == null || cacheDir.isEmpty()) {
                return getCacheDir();
            }
        }
        return Paths.get(cacheDir);
    }

    /**
     * Utility function to get DJL cache directory.
     *
     * @return DJL cache directory
     */
    public static Path getCacheDir() {
        String cacheDir = System.getProperty(MXNET_CACHE_DIR);
        if (cacheDir == null || cacheDir.isEmpty()) {
            cacheDir = System.getenv(MXNET_CACHE_DIR);
            if (cacheDir == null || cacheDir.isEmpty()) {
                Path dir = Paths.get(System.getProperty("user.home"));
                if (!Files.isWritable(dir)) {
                    dir = Paths.get(System.getProperty("java.io.tmpdir"));
                }
                return dir.resolve("mxnet.java_package");
            }
        }
        return Paths.get(cacheDir);
    }

}
