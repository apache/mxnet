/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.apache.mxnet.util;

import java.util.Locale;

/** A class containing utility methods. */
public final class FilenameUtils {

    private FilenameUtils() {}

    /**
     * Returns the type of the file.
     *
     * @param fileName the file name
     * @return the type of the file
     */
    public static String getFileType(String fileName) {
        fileName = fileName.toLowerCase(Locale.ROOT);
        if (fileName.endsWith(".zip")) {
            return "zip";
        } else if (fileName.endsWith(".tgz")
                || fileName.endsWith(".tar.gz")
                || fileName.endsWith(".tar.z")) {
            return "tgz";
        } else if (fileName.endsWith(".tar")) {
            return "tar";
        } else if (fileName.endsWith(".gz") || fileName.endsWith(".z")) {
            return "gzip";
        } else {
            return "";
        }
    }

    /**
     * Returns if the the file is an archive file.
     *
     * @param fileName the file name
     * @return the type of the file
     */
    public static boolean isArchiveFile(String fileName) {
        String fileType = getFileType(fileName);
        return "tgz".equals(fileType) || "zip".equals(fileType) || "tar".equals(fileType);
    }

    /**
     * Returns the name of the file without file extension.
     *
     * @param name the file name
     * @return the name of the file without file extension
     */
    public static String getNamePart(String name) {
        String lowerCase = name.toLowerCase(Locale.ROOT);
        if (lowerCase.endsWith(".tar.gz")) {
            return name.substring(0, name.length() - 7);
        } else if (name.endsWith(".tar.z")) {
            return name.substring(0, name.length() - 6);
        } else if (name.endsWith(".tgz") || name.endsWith(".zip") || name.endsWith(".tar")) {
            return name.substring(0, name.length() - 4);
        } else if (name.endsWith(".gz")) {
            return name.substring(0, name.length() - 3);
        } else if (name.endsWith(".z")) {
            return name.substring(0, name.length() - 2);
        }
        return name;
    }

    /**
     * Returns the file name extension of the file.
     *
     * @param fileName the file name
     * @return the file name extension
     */
    public static String getFileExtension(String fileName) {
        int pos = fileName.lastIndexOf('.');
        if (pos > 0) {
            return fileName.substring(pos + 1);
        }
        return "";
    }
}
