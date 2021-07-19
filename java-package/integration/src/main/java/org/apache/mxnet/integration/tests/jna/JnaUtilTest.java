package org.apache.mxnet.integration.tests.jna;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.DataType;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.training.initializer.Initializer;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void doForwardTest() {
        // TODO: replace the Path of model with soft decoding
        try (
                MxResource base = BaseMxResource.getSystemMxResource()
                ) {
            Symbol symbol  = Symbol.loadFromFile(base,
                    "/Users/cspchen/.djl.ai/cache/repo/model/cv/image_classification/ai/djl/mxnet/mlp/mnist/0.0.1/mlp-symbol.json");
            MxSymbolBlock block = new MxSymbolBlock(base, symbol);
            Device device = Device.defaultIfNull();
            MxNDList mxNDArray = JnaUtils.loadNdArray(
                    base,
                    Paths.get("/Users/cspchen/.djl.ai/cache/repo/model/cv/image_classification/ai/djl/mxnet/mlp/mnist/0.0.1/mlp-0000.params"),
                    Device.defaultIfNull(null));

            // load parameters
            List<Parameter> parameters = block.getAllParameters();
            Map<String, Parameter> map = new LinkedHashMap<>();
            parameters.forEach(p -> map.put(p.getName(), p));

            for (MxNDArray nd : mxNDArray) {
                String key = nd.getName();
                if (key == null) {
                    throw new IllegalArgumentException("Array names must be present in parameter file");
                }

                String paramName = key.split(":", 2)[1];
                Parameter parameter = map.remove(paramName);
                parameter.setArray(nd);
            }
            block.setInputNames(new ArrayList<>(map.keySet()));

            MxNDArray arr = MxNDArray.create(base, new Shape(1, 28, 28), device).ones();
            block.forward(new ParameterStore(base, false, device), new MxNDList(arr), false, new PairList<>(), device);
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
