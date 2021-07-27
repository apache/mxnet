# Java Package for MXNet 2.0

## Requirements

## Install

## Scripts
- customize mxnet library path  
```bash
export MXNET_LIBRARY_PATH=//anaconda3/lib/python3.8/site-packages/mxnet/
```


## Tests  
Test case for a rough inference run with MXNet model  
```bash
./gradlew :integration:run  
```

## Example

```java
try (MxResource base = BaseMxResource.getSystemMxResource())
        {
        Model model = Model.loadModel(Item.MLP);
//            Model model = Model.loadModel("test", Paths.get("/Users/cspchen/mxnet.java_package/cache/repo/test-models/mlp.tar.gz/mlp/"));
        Predictor<NDList, NDList> predictor = model.newPredictor();
        NDArray input = NDArray.create(base, new Shape(1, 28, 28)).ones();
        NDList inputs = new NDList();
        inputs.add(input);
        NDList result = predictor.predict(inputs);
        NDArray expected =  NDArray.create(
        base,
        new float[]{4.93476f, -0.76084447f, 0.37713608f, 0.6605506f, -1.3485785f, -0.8736369f
        , 0.018061712f, -1.3274033f, 1.0609543f, 0.24042489f}, new Shape(1, 10));
        Assertions.assertAlmostEquals(result.get(0), expected);

        } catch (IOException e) {
        logger.error(e.getMessage(), e);
        }
```