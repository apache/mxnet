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
