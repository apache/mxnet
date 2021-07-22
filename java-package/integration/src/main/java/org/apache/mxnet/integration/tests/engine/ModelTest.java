package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Predictor;
import org.apache.mxnet.integration.tests.jna.JnaUtilTest;
import org.apache.mxnet.integration.util.Assertions;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.repository.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;

public class ModelTest {
    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void modelLoadAndPredictTest() {
        try (MxResource base = BaseMxResource.getSystemMxResource())
        {
            MxModel mxModel = MxModel.loadModel(Item.MLP);
//            MxModel mxModel = MxModel.loadModel("trest", Paths.get("/Users/cspchen/mxnet.java_package/cache/repo/test-models/mlp.tar.gz/mlp/"));
            Predictor<MxNDList, MxNDList> predictor = mxModel.newPredictor();
            MxNDArray input = MxNDArray.create(base, new Shape(1, 28, 28)).ones();
            MxNDList inputs = new MxNDList();
            inputs.add(input);
            MxNDList result = predictor.predict(inputs);
            MxNDArray expected =  MxNDArray.create(
                    base,
                    new float[]{4.93476f, -0.76084447f, 0.37713608f, 0.6605506f, -1.3485785f, -0.8736369f
                            , 0.018061712f, -1.3274033f, 1.0609543f, 0.24042489f}, new Shape(1, 10));
            Assertions.assertAlmostEquals(result.get(0), expected);

        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
    }
}
