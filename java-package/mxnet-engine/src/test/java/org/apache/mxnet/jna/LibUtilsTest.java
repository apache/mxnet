package org.apache.mxnet.jna;

import org.junit.jupiter.api.Test;

import java.nio.IntBuffer;

import static org.junit.jupiter.api.Assertions.*;

class LibUtilsTest {


    @Test
    void loadLibraryFromCustomizePath() {

        MxnetLibrary lib = LibUtils.loadLibrary();
        IntBuffer version = IntBuffer.allocate(1);
        int ret = lib.MXGetVersion(version);
        assertEquals(0, ret);
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