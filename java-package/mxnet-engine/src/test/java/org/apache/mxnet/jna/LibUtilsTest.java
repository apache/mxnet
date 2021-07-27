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

import com.sun.jna.Pointer;
import java.nio.IntBuffer;
import org.testng.annotations.Test;

public class LibUtilsTest {

    @Test
    void loadLibraryFromCustomizePathTest() {
        MxnetLibrary lib = LibUtils.loadLibrary();
        IntBuffer version = IntBuffer.allocate(1);
        int ret = lib.MXGetVersion(version);
        assert ret == 0;
    }

    @Test
    void createSymbolFromFileTest() {
        try {
            Pointer p =
                    JnaUtils.createSymbolFromFile(
                            "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json");
            System.out.println("catch");
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    //    @Test
    //    void loadLibraryFromJavaLibraryPath() {
    //
    //        String libPath = System.getProperty("java.library.path");
    //        String absolutePath = "";
    //        if (libPath != null) {
    //            absolutePath = LibUtils.findLibraryInPath(libPath);
    //        }
    //
    //    }
}
