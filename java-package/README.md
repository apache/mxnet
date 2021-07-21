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
            MxModel mxModel = MxModel.loadModel(Item.MLP);
//            MxModel.loadModel(Item.MLP.getName(), Paths.get(Item.MLP.getUrl());
            Predictor<MxNDList, MxNDList> predictor = mxModel.newPredictor();
            MxNDArray input = MxNDArray.create(base, new Shape(1, 28, 28)).ones();
            MxNDList inputs = new MxNDList();
            inputs.add(input);
            MxNDList result = predictor.predict(inputs);
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
}
```