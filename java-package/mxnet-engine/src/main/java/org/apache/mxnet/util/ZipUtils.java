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

package org.apache.mxnet.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/** Utilities for working with zip files. */
public final class ZipUtils {

    private ZipUtils() {}

    /**
     * Unzips an input stream to a given path.
     *
     * @param is the input stream to unzip
     * @param dest the path to store the unzipped files
     * @throws IOException for failures to unzip the input stream and create files in the dest path
     */
    public static void unzip(InputStream is, Path dest) throws IOException {
        ZipInputStream zis = new ZipInputStream(is);
        ZipEntry entry;
        while ((entry = zis.getNextEntry()) != null) {
            String name = entry.getName();
            if (name.contains("..")) {
                throw new IOException("Malicious zip entry: " + name);
            }
            Path file = dest.resolve(name).toAbsolutePath();
            if (entry.isDirectory()) {
                Files.createDirectories(file);
            } else {
                Path parentFile = file.getParent();
                if (parentFile == null) {
                    throw new AssertionError(
                            "Parent path should never be null: " + file.toString());
                }
                Files.createDirectories(parentFile);
                Files.copy(zis, file, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    /**
     * Zips an input directory to a given file.
     *
     * @param src the input directory to zip
     * @param dest the path to store the zipped files
     * @param includeFolderName if include the source directory name in the zip entry
     * @throws IOException for failures to zip the input directory
     */
    public static void zip(Path src, Path dest, boolean includeFolderName) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(dest))) {
            Path root = includeFolderName ? src.getParent() : src;
            if (root == null) {
                throw new AssertionError("Parent folder should not be null.");
            }
            addToZip(root, src, zos);
        }
    }

    private static void addToZip(Path root, Path file, ZipOutputStream zos) throws IOException {
        Path relative = root.relativize(file);
        String name = relative.toString();
        if (Files.isDirectory(file)) {
            if (!name.isEmpty()) {
                ZipEntry entry = new ZipEntry(name + '/');
                zos.putNextEntry(entry);
            }
            File[] files = file.toFile().listFiles();
            if (files != null) {
                for (File f : files) {
                    addToZip(root, f.toPath(), zos);
                }
            }
        } else if (Files.isRegularFile(file)) {
            if (name.isEmpty()) {
                name = file.toFile().getName();
            }
            ZipEntry entry = new ZipEntry(name);
            zos.putNextEntry(entry);
            Files.copy(file, zos);
        }
    }
}
