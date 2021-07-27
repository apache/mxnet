package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.exception.JnaCallException;
import org.apache.mxnet.jna.JnaUtils;
import org.testng.annotations.Test;

import java.nio.file.Paths;

public class SymbolTest {

    @Test
    public void loadAndCloseTest() {
        try (Symbol symbol =
                     Symbol.loadSymbol(BaseMxResource.getSystemMxResource(),
                             Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json"))) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());
        } catch (JnaCallException e) {
            e.printStackTrace();
        }
    }
}
