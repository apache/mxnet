package org.apache.mxnet.integration.tests.jna;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.apache.mxnet.training.ParameterStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;


public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void createCachedOpTest() {
        try (
                MxResource base = BaseMxResource.getSystemMxResource()
                ) {
            Symbol symbol  = Symbol.loadFromFile(base,
                    "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json");
            MxSymbolBlock block = new MxSymbolBlock(base, symbol);

            MxNDList mxNDArray = JnaUtils.loadNdArray(
                    base,
                    Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params"),
                    Device.defaultIfNull(null));

            CachedOp cachedOp = JnaUtils.createCachedOp(block, base, false);
            cachedOp.forward(new ParameterStore(base, false, Device.defaultIfNull(null)), mxNDArray, false);
            System.out.println(base.getSubResource().size());
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
            throw e;
        }
        System.out.println("Ok");
        System.out.println(BaseMxResource.getSystemMxResource().getSubResource().size());

    }

    @Test
    public void createNdArray() {
        try {
            try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                MxNDArray intArray = MxNDArray.create(base, new int[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2}, new Shape(3, 4));
                MxNDArray floatArray = MxNDArray.create(base, new float[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2}, new Shape(3, 4));
                MxNDArray doubleArray = MxNDArray.create(base, new double[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2}, new Shape(3, 4));
                MxNDArray longArray = MxNDArray.create(base, new long[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2}, new Shape(3, 4));
                MxNDArray booleanArray = MxNDArray.create(base, new boolean[]{true, false, false, true, true, true, true, false, false, true, true, true}, new Shape(3, 4));
                MxNDArray byteArray = MxNDArray.create(base, new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new Shape(3, 4));
                MxNDArray intArray2 = MxNDArray.create(base, new int[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2});
                MxNDArray floatArray2 = MxNDArray.create(base, new float[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2});
                MxNDArray doubleArray2 = MxNDArray.create(base, new double[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2});
                MxNDArray longArray2 = MxNDArray.create(base, new long[]{1, 23, 2, 32, 3, 23, 1234, 1, 24, 2, 3, 1, 2});
                MxNDArray booleanArray2 = MxNDArray.create(base, new boolean[]{true, false, false, true, true, true, true, false, false, true, true, true});
                MxNDArray byteArray2 = MxNDArray.create(base, new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
                Integer[] ndArrayInt = (Integer[]) intArray.toArray();
                // Float -> Double
                double[] floats = Arrays.stream(floatArray.toArray()).mapToDouble(Number::floatValue).toArray();
                Double[] ndArrayDouble = (Double[]) doubleArray.toArray();
                Long[] ndArrayLong = (Long[]) longArray.toArray();
                boolean[] ndArrayBoolean = booleanArray.toBooleanArray();
                byte[] ndArrayByte = byteArray.toByteArray();

                Integer[] ndArrayInt2 = (Integer[]) intArray2.toArray();
                // Float -> Double
                double[] floats2 = Arrays.stream(floatArray2.toArray()).mapToDouble(Number::floatValue).toArray();
                Double[] ndArrayDouble2 = (Double[]) doubleArray2.toArray();
                Long[] ndArrayLong2 = (Long[]) longArray2.toArray();
                boolean[] ndArrayBoolean2 = booleanArray2.toBooleanArray();
                byte[] ndArrayByte2 = byteArray2.toByteArray();
            } catch (Exception e) {
                logger.error(e.getMessage());
                e.printStackTrace();
                throw e;
            }
            BaseMxResource base = BaseMxResource.getSystemMxResource();
//            assert base.getSubResource().size() == 0;
        } catch (Exception e) {
            logger.error(e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void loadNdArray() {

        try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                MxNDList mxNDArray = JnaUtils.loadNdArray(
                        base,
                        Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params"),
                        Device.defaultIfNull(null));

            System.out.println(base.getSubResource().size());
        }
        System.out.println(BaseMxResource.getSystemMxResource().getSubResource().size());

    }
}
