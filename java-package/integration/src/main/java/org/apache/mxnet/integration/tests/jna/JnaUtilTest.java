package org.apache.mxnet.integration.tests.jna;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.testng.annotations.Test;

public class JnaUtilTest {

    @Test
    public void createCachedOpTest() {
        try (
                MxResource base = BaseMxResource.getSystemMxResource();
                Symbol symbol  = Symbol.loadFromFile(base,
                        "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json");
                MxSymbolBlock block = new MxSymbolBlock(base, symbol);
                CachedOp cachedOp = JnaUtils.createCachedOp(block, base, false)
                ) {
            cachedOp.getUid();
        }
        System.out.println("Ok");

    }

}
