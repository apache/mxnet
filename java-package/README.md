<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

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