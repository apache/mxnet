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

## How to compose SSD network on top of mainstream classification networks

1. Have the base network ready in this directory as `name.py`, such as `inceptionv3.py`.
2. Add configuration to `symbol_factory.py`, an example would be:
```
if network == 'vgg16_reduced':
    if data_shape >= 448:
        from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
        num_filters = [512, -1, 512, 256, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2, 1]
        pads = [-1, -1, 1, 1, 1, 1, 1]
        sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
            [.75, .8216], [.9, .9721]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        normalizations = [20, -1, -1, -1, -1, -1, -1]
        steps = [] if data_shape != 512 else [x / 512.0 for x in
            [8, 16, 32, 64, 128, 256, 512]]
    else:
        from_layers = ['relu4_3', 'relu7', '', '', '', '']
        num_filters = [512, -1, 512, 256, 256, 256]
        strides = [-1, -1, 2, 2, 1, 1]
        pads = [-1, -1, 1, 1, 0, 0]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = [20, -1, -1, -1, -1, -1]
        steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
    return locals()
elif network == 'inceptionv3':
    from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
    num_filters = [-1, -1, 512, 256, 256, 128]
    strides = [-1, -1, 2, 2, 2, 2]
    pads = [-1, -1, 1, 1, 1, 1]
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5], [1,2,.5]]
    normalizations = -1
    steps = []
    return locals()
```
Here `from_layers` indicate the feature layer you would like to extract from the base network.
`''` indicate that we want add extra new layers on top of the last feature layer,
and the number of filters must be specified in `num_filters`. Similarly, `strides` and `pads`
are required to compose these new layers. `sizes` and `ratios` are the parameters controlling
the anchor generation algorithm. `normalizations` is used to normalize and rescale feature if
not `-1`. `steps`: optional, used to calculate the anchor sliding steps.

3. Train or test with arguments `--network name --data-shape xxx --pretrained pretrained_model`
