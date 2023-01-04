---
layout: page_api
title: SSD Inference
permalink: /api/java/docs/tutorials/ssd_inference
is_tutorial: true
tag: java
---
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

# Multi Object Detection using pre-trained SSD Model via Java Inference APIs

This tutorial shows how to use MXNet Java Inference APIs to run inference on a pre-trained Single Shot Detector (SSD) Model.

The SSD model is trained on the Pascal VOC 2012 dataset. The network is a SSD model built on Resnet50 as the base network to extract image features. The model is trained to detect the following entities (classes): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']. For more details about the model, you can refer to the [MXNet SSD example](https://github.com/apache/mxnet/tree/master/example/ssd).

## Prerequisites

To complete this tutorial, you need the following:
* [MXNet Java Setup on IntelliJ IDEA](mxnet_java_on_intellij) (Optional)
* [wget](https://www.gnu.org/software/wget/) To download model artifacts
* SSD Model artifacts
    * Use the following script to get the SSD Model files :
```bash
data_path=/tmp/resnet50_ssd
mkdir -p "$data_path"
wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json -P $data_path
wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params -P $data_path
wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt -P $data_path
```
* Test images  : A few sample images to run inference on.
    * Use the following script to download sample images :
```bash
image_path=/tmp/resnet50_ssd/images
mkdir -p "$image_path"
cd $image_path
wget https://cloud.githubusercontent.com/assets/3307514/20012567/cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg -O dog.jpg
wget https://cloud.githubusercontent.com/assets/3307514/20012563/cbb41382-a27d-11e6-92a9-18dab4fd1ad3.jpg -O person.jpg
```

Alternately, you can get the entire SSD Model artifacts + images in one single script from the MXNet Repository by running [get_ssd_data.sh script](https://github.com/apache/mxnet/blob/master/scala-package/examples/scripts/infer/objectdetector/get_ssd_data.sh)

## Time to code!
1\. Following the [MXNet Java Setup on IntelliJ IDEA](mxnet_java_on_intellij) tutorial, in the same project `JavaMXNet`, create a new empty class called : `ObjectDetectionTutorial.java`.

2\. In the `main` function of `ObjectDetectionTutorial.java` define the downloaded model path and the image data paths. This is the same path where we downloaded the model artifacts and images in a previous step.

```java
String modelPathPrefix = "/tmp/resnet50_ssd/resnet50_ssd_model";
String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";
```

3\. We can run the inference code in this example on either CPU or GPU (if you have a GPU backed machine) by choosing the appropriate context.

```java

List<Context> context = getContext();
...

private static List<Context> getContext() {
List<Context> ctx = new ArrayList<>();
ctx.add(Context.cpu()); // Choosing CPU Context here

return ctx;
}
```

4\. To provide an input to the model, define the input shape to the model and the Input Data Descriptor (DataDesc) as shown below :

```java
Shape inputShape = new Shape(new int[] {1, 3, 512, 512});
List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));
```

The input shape can be interpreted as follows : The input has a batch size of 1, with 3 RGB channels in the image, and the height and width of the image is 512 each.

5\. To run an actual inference on the given image, add the following lines to the `ObjectDetectionTutorial.java` class :

```java
BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
List<List<ObjectDetectorOutput>> output = objDet.imageObjectDetect(img, 3); // Top 3 objects detected will be returned
```

6\. Let's piece all of the above steps together by showing the final contents of the `ObjectDetectionTutorial.java`.

```java
package mxnet;

import org.apache.mxnet.infer.javaapi.ObjectDetector;
import org.apache.mxnet.infer.javaapi.ObjectDetectorOutput;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.Shape;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ObjectDetectionTutorial {

    public static void main(String[] args) {

        String modelPathPrefix = "/tmp/resnet50_ssd/resnet50_ssd_model";

        String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";

        List<Context> context = getContext();

        Shape inputShape = new Shape(new int[] {1, 3, 512, 512});

        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));

        BufferedImage img = ObjectDetector.loadImageFromFile(inputImagePath);
        ObjectDetector objDet = new ObjectDetector(modelPathPrefix, inputDescriptors, context, 0);
        List<List<ObjectDetectorOutput>> output = objDet.imageObjectDetect(img, 3);

        printOutput(output, inputShape);
    }


    private static List<Context> getContext() {
        List<Context> ctx = new ArrayList<>();
        ctx.add(Context.cpu());

        return ctx;
    }

    private static void printOutput(List<List<ObjectDetectorOutput>> output, Shape inputShape) {

        StringBuilder outputStr = new StringBuilder();

        int width = inputShape.get(3);
        int height = inputShape.get(2);

        for (List<ObjectDetectorOutput> ele : output) {
            for (ObjectDetectorOutput i : ele) {
                outputStr.append("Class: " + i.getClassName() + "\n");
                outputStr.append("Probabilties: " + i.getProbability() + "\n");

                List<Float> coord = Arrays.asList(i.getXMin() * width,
                        i.getXMax() * height, i.getYMin() * width, i.getYMax() * height);
                StringBuilder sb = new StringBuilder();
                for (float c: coord) {
                    sb.append(", ").append(c);
                }
                outputStr.append("Coord:" + sb.substring(2)+ "\n");
            }
        }
        System.out.println(outputStr);

    }
}
```

7\. To compile and run this code, change directories to this project's root folder, then run the following:
```bash
mvn clean install dependency:copy-dependencies
```

The build generates a new jar file in the `target` folder called `javaMXNet-1.0-SNAPSHOT.jar`.

To run the ObjectDetectionTutorial.java use the following command from the project's root folder.
```bash
java -cp "target/javaMXNet-1.0-SNAPSHOT.jar:target/dependency/*" mxnet.ObjectDetectionTutorial
```

You should see a similar output being generated for the dog image that we used:
```bash
Class: car
Probabilties: 0.99847263
Coord:312.21335, 72.02908, 456.01443, 150.66176
Class: bicycle
Probabilties: 0.9047381
Coord:155.9581, 149.96365, 383.83694, 418.94516
Class: dog
Probabilties: 0.82268167
Coord:83.82356, 179.14001, 206.63783, 476.78754
```

![dog_1](https://cloud.githubusercontent.com/assets/3307514/20012567/cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg)

The results returned by the inference call translate into the regions in the image where the model detected objects.

![dog_2](https://cloud.githubusercontent.com/assets/3307514/19171063/91ec2792-8be0-11e6-983c-773bd6868fa8.png)

## Next Steps
For more information about MXNet Java resources, see the following:

* [Java Inference API]({{'/api/java'|relative_url}})
* [Java Inference Examples](https://github.com/apache/mxnet/tree/master/scala-package/examples/src/main/java/org/apache/mxnetexamples/javaapi/infer)
* [MXNet Tutorials Index]({{'/api'|relative_url}})
