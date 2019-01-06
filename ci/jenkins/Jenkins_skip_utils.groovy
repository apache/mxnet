// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

def only_non_compile_changed(file_list) {
    // Returns true if only files are changed that do not affect compilation output
    println "Changed files: $file_list"
    // Check if all files belong to the whitelisted extensions
    String[] non_compile_extensions = ['md', 'txt']

    non_compile_files = file_list.findAll { path ->
        path_elements = path.split("\\.")
        if (path_elements.length == 0) {
            return false
        }

        path_elements[-1] in extensions
    }
    println "Noncompile files: $non_compile_files"
    if (non_compile_files.length != file_list.length) {
        return false
    }

    // Check if any file is in a blacklisted path
    String[] blacklisted_paths = ['docs/tutorials']

    non_blacklisted_paths = file_list.findAll { path ->
        for (b_path in blacklisted_paths) {
            if (path.startsWith(b_path)) {
                return false
            }
        }
        return true
    }
    println "nonblacklisted files: $non_blacklisted_paths"
    if (non_blacklisted_paths.length != file_list.length) {
        return false
    }
    return true
}
return this