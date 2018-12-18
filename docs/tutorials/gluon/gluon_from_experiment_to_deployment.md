
# Gluon: from experiment to deployment, an end to end example

## Overview
MXNet Gluon API comes with a lot of great features and it can provide you everything you need from experiment to deploy the model. In this tutorial, we will walk you through a common use case on how to build a model using gluon, train it on your data, and deploy it for inference.

Let's say you need to build a service that provides flower species recognition. A common use case is, you don't have enough data to train a good model. In such cases we use a technique called Transfer Learning.
In Transfer Learning we make use of a pre-trained model that solves a related task but is trained on a very large standard dataset such as ImageNet from a different domain, we utilize the knowledge in this pre-trained model to perform a new task at hand.

Gluon provides State of the Art models for many of the standard tasks such as Classification, Object Detection, Segmentation, etc. In this tutorial we will use the pre-trained model [ResNet50 V2](https://arxiv.org/abs/1603.05027) trained on ImageNet dataset, this model achieves 77.11% top-1 accuracy on ImageNet, we seek to transfer as much knowledge as possible for our task of recognizing different species of Flowers.

In this tutorial we will show you the steps to load pre-trained model from Gluon, tweak the model according to your need, fine-tune the model on your small dataset, and finally deploy the trained model to integrate with your service.




## Prerequisites

To complete this tutorial, you need:

- [Build MXNet from source](https://mxnet.incubator.apache.org/install/ubuntu_setup.html#build-mxnet-from-source) with Python(Gluon) and C++ Packages
- Learn the basics about Gluon with [A 60-minute Gluon Crash Course](https://gluon-crash-course.mxnet.io/)
- Learn the basics about [MXNet C++ API](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)


## The Data

We will use the [Oxford 102 Category Flower Dateset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) as an example to show you the steps. You can use this [script](https://github.com/Arsey/keras-transfer-learning-for-oxford102/blob/master/bootstrap.py) to download and organize your data into train, test, and validation sets. Simply import it and run:



```python
data_util_file = "oxford_102_flower_dataset.py"
base_url = "https://raw.githubusercontent.com/roywei/incubator-mxnet/gluon_tutorial/docs/tutorial_utils/data/{}?raw=true"
mx.test_utils.download(base_url.format(data_util_file), fname=data_util_file)
import oxford_102_flower_dataset

# download and move data to train, test, valid folders
path = './data'
oxford_102_flower_dataset.get_data(path)
```

Now your data will be organized into the following format, all the images belong to the same category will be put together
```bash
data
|--train
|   |-- 0
|   |   |-- image_06736.jpg
|   |   |-- image_06741.jpg
...
|   |-- 1
|   |   |-- image_06755.jpg
|   |   |-- image_06899.jpg
...
|-- test
|   |-- 0
|   |   |-- image_00731.jpg
|   |   |-- image_0002.jpg
...
|   |-- 1
|   |   |-- image_00036.jpg
|   |   |-- image_05011.jpg

```

## Training using Gluon

### Define Hyper-parameters

Now let's first import necessary packages:


```python
import math
import os
import time
from multiprocessing import cpu_count

import mxnet as mx
from mxnet import autograd
from mxnet import gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo.vision import resnet50_v2
```

and define the hyper-parameters we will use for fine-tuning, we will use [MXNet learning rate scheduler](https://mxnet.incubator.apache.org/tutorials/gluon/learning_rate_schedules.html) to adjust learning rates during training.


```python
classes = 102
epochs = 40
lr = 0.001
per_device_batch_size = 32
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
# learning rate change at following epochs
lr_epochs = [10, 20, 30]

num_gpus = mx.context.num_gpus()
num_workers = cpu_count()
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)
```

Before the training we will apply data augmentations on training images. It's making minor alterations on training images and our model will consider them as distinct images. This can be very useful for finetuning on relatively small dataset and help improve the model. We can use Gluon [DataSet API](https://mxnet.incubator.apache.org/tutorials/gluon/datasets.html), [DataLoader API](https://mxnet.incubator.apache.org/tutorials/gluon/datasets.html), and [Transform API](https://mxnet.incubator.apache.org/tutorials/gluon/data_augmentation.html) to load the images and apply the follwing data augmentation:
1. Randomly crop the image and resize it to 224x224
2. Randomly flip the image horizontally
3. Randomly jitter color and add noise
4. Transpose the data from height*width*num_channels to num_channels*height*width, and map values from [0, 255] to [0, 1]
5. Normalize with the mean and standard deviation from the ImageNet dataset.



```python
jitter_param = 0.4
lighting_param = 0.1

training_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'valid')
test_path = os.path.join(path, 'test')

# loading the data and apply pre-processing(transforms) on images
train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

### Loading pre-trained model


We will use pre-trained ResNet50_v2 model which was pre-trained on the [ImageNet Dataset](http://www.image-net.org/) with 1000 classes. All you need to do is re-define the last softmax layer and specify the number of classes to be 102 in our case and initialize the parameters. You can also add layers to the network according to your needs.

Before we go to training, one unique feature Gluon offers is hybridization. It allows you to convert your imperative code to static symbolic graph which is much more efficient to execute. There are two main benefit of hybridizing your model: better performance and easier serialization for deployment. The best part is it's as simple as just calling `net.hybridize()`. To know more about Gluon hybridization, please follow our [tutorials](https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html).



```python
# load pre-trained resnet50_v2 from model zoo
finetune_net = resnet50_v2(pretrained=True, ctx=ctx)

# change last softmax layer since number of classes are different
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
# hybridize for better performance
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
    'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

### Fine-tuning model on your custom dataset

Now let's define the test metrics and start fine-tuning.



```python
def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, (data, label) in enumerate(val_data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)
        outputs = [net(x) for x in data]
        metric.update(label, outputs)
    return metric.get()


num_batch = len(train_data)
iteration_idx = 1

# setup learning rate scheduler
iterations_per_epoch = math.ceil(num_batch)
# learning rate change at following steps
lr_steps = [epoch * iterations_per_epoch for epoch in lr_epochs]
schedule = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=lr_factor, base_lr=lr)

# start with epoch 1 for easier learning rate calculation
for epoch in range(1, epochs + 1):

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, (data, label) in enumerate(train_data):
        # get the images and labels
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)
        with autograd.record():
            outputs = [finetune_net(x) for x in data]
            loss = [softmax_cross_entropy(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        lr = schedule(iteration_idx)
        trainer.set_learning_rate(lr)
        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
        metric.update(label, outputs)
        iteration_idx += 1

    _, train_acc = metric.get()
    train_loss /= num_batch
    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | learning-rate: %.3E | time: %.1f' %
          (epoch, train_acc, train_loss, val_acc, trainer.learning_rate, time.time() - tic))

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
```

Following is the training result:
```bash
[Epoch 40] Train-acc: 0.945, loss: 0.354 | Val-acc: 0.955 | learning-rate: 4.219E-04 | time: 17.8
[Finished] Test-acc: 0.952
```
We trained the model using a [AWS P3.8XLarge instance](https://aws.amazon.com/ec2/instance-types/p3/) with 4 Tesla V100	GPUs. We were able to reach a test accuracy of 95.2% with 40 epochs in around 12 minutes. This was really fast because our model was pre-trained on a much larger dataset, ImageNet, with around 1.3 million images. It worked really well to capture features on our small dataset.


### Save fine-tuned model


We now have a trained our custom model. This can be serialized into model files using the export function. The export function will export the model architecture into a `.json` file and model parameters into a `.params` file.



```python
finetune_net.export("flower-recognition", epoch=epochs)

```

export in this case creates `flower-recognition-symbol.json` and `flower-recognition-0020.params` in the current directory. They can be used for model deployment in the next section.


## Load and inference using C++ API

MXNet provide various useful tools and interfaces for deploying your model for inference. For example, you can use [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) to start a service and host your trained model easily. Besides that, you can also use MXNet's different language APIs to integrate your model with your existing service. We provide [Java](https://mxnet.incubator.apache.org/api/java/index.html), [Scala](https://mxnet.incubator.apache.org/api/scala/index.html), and [C++](https://mxnet.incubator.apache.org/api/c++/index.html) APIs. In this tutorial, we will focus on the C++ API, for more details, please refer to the [C++ Inference Example](https://github.com/leleamol/incubator-mxnet/tree/inception-example/cpp-package/example/inference).


### Setup MXNet C++ API
To use C++ API in MXNet, you need to build MXNet from source with C++ package. Please follow the [built from source guide](https://mxnet.incubator.apache.org/install/ubuntu_setup.html), and [C++ Package documentation](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)
to enable C++ API.
In summary you just need to build MXNet from source with `USE_CPP_PACKAGE` flag set to 1 using `make -j USE_CPP_PACKAGE=1`.

### Write Predictor in C++
Now let's write prediction code in C++.  We will use a Predictor Class to do the following jobs:
1. Load the pre-trained model,
2. Load the parameters of pre-trained model,
3. Load the image to be classified in to NDArray.
4. Run the forward pass and predict the input image.

```cpp
class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              bool gpu_context_type = false,
              const std::string& synset_file = "",
              const std::string& mean_image_file = "");
    void PredictImage(const std::string& image_file);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadSynset(const std::string& synset_file);
    NDArray LoadInputImage(const std::string& image_file);
    void LoadMeanImageData();
    void LoadDefaultMeanImageData();
    void NormalizeInput(const std::string& mean_image_file);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    NDArray mean_img;
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::vector<std::string> output_labels;
    Symbol net;
    Executor *executor;
    Shape input_shape;
    NDArray mean_image_data;
    NDArray std_dev_image_data;
    Context global_ctx = Context::cpu();
    std::string mean_image_file;
};
```

### Load network symbol and parameters

In the Predictor constructor, you need a few information including paths to saved json and param files. After that add the following two methods to load the network and its parameters.

```cpp
/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
  if (!FileExists(model_json_file)) {
    LG << "Model file " << model_json_file << " does not exist";
    throw std::runtime_error("Model file does not exist");
  }
  LG << "Loading the model from " << model_json_file << std::endl;
  net = Symbol::Load(model_json_file);
}


/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
  if (!FileExists(model_parameters_file)) {
    LG << "Parameter file " << model_parameters_file << " does not exist";
    throw std::runtime_error("Model parameters does not exist");
  }
  LG << "Loading the model parameters from " << model_parameters_file << std::endl;
  std::map<std::string, NDArray> parameters;
  NDArray::Load(model_parameters_file, 0, &parameters);
  for (const auto &k : parameters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(global_ctx);
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(global_ctx);
    }
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}
```

### Load Input Image

Now let's add a method to load the input image we want to predict and converts it to NDArray for prediction.
```cpp
NDArray Predictor::LoadInputImage(const std::string& image_file) {
  if (!FileExists(image_file)) {
    LG << "Image file " << image_file << " does not exist";
    throw std::runtime_error("Image file does not exist");
  }
  LG << "Loading the image " << image_file << std::endl;
  std::vector<float> array;
  cv::Mat mat = cv::imread(image_file);
  /*resize pictures to (224, 224) according to the pretrained model*/
  int height = input_shape[2];
  int width = input_shape[3];
  int channels = input_shape[1];
  cv::resize(mat, mat, cv::Size(height, width));
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        array.push_back(static_cast<float>(mat.data[(i * height + j) * 3 + c]));
      }
    }
  }
  NDArray image_data = NDArray(input_shape, global_ctx, false);
  image_data.SyncCopyFromCPU(array.data(), input_shape.Size());
  NDArray::WaitAll();
  return image_data;
}
```

### Run inference

Finally, let's run the inference. It's basically using MXNet executor to do a forward pass. To run predictions on multiple images, you can load the images in a list of NDArrays and run prediction in batches. Note that the Predictor class may not be thread safe, calling it in multi-threaded environments was not tested. To utilize multi-threaded prediction, you need to use the C predict API, please follow the [C predict example](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp).

```cpp
void Predictor::PredictImage(const std::string& image_file) {
  // Load the input image
  NDArray image_data = LoadInputImage(image_file);

  // Normalize the image
  image_data.Slice(0, 1) -= mean_image_data;

  LG << "Running the forward pass on model to predict the image";
  /*
   * The executor->arg_arrays represent the arguments to the model.
   *
   * Copying the image_data that contains the NDArray of input image
   * to the arg map of the executor. The input is stored with the key "data" in the map.
   *
   */
  image_data.CopyTo(&(executor->arg_dict()["data"]));
  NDArray::WaitAll();

  // Run the forward pass.
  executor->Forward(false);

  // The output is available in executor->outputs.
  auto array = executor->outputs[0].Copy(global_ctx);
  NDArray::WaitAll();

  /*
   * Find out the maximum accuracy and the index associated with that accuracy.
   * This is done by using the argmax operator on NDArray.
   */
  auto predicted = array.ArgmaxChannel();
  NDArray::WaitAll();

  int best_idx = predicted.At(0, 0);
  float best_accuracy = array.At(0, best_idx);

  if (output_labels.empty()) {
    LG << "The model predicts the highest accuracy of " << best_accuracy << " at index "
       << best_idx;
  } else {
    LG << "The model predicts the input image to be a [" << output_labels[best_idx]
       << " ] with Accuracy = " << best_accuracy << std::endl;
  }
}
```

### Compile and Run Inference Code

You can find the full code [here](https://github.com/leleamol/incubator-mxnet/blob/inception-example/cpp-package/example/inference/inception_inference.cpp)
, and to compile it use this [Makefile](https://github.com/leleamol/incubator-mxnet/blob/inception-example/cpp-package/example/inference/Makefile)

Now you will be able to compile the run inference, just do `make all` and pass the parameters as follows

```bash
make all
LD_LIBRARY_PATH=../incubator-mxnet/lib/ ./inception_inference --symbol "flower-recognition-symbol.json" --params "flower-recognition-0020.params" --image ./data/test/0/image_06736.jpg
```

Then it will predict your iamge

```bash
[22:26:49] inception_inference.cpp:128: Loading the model from flower-recognition-symbol.json

[22:26:49] inception_inference.cpp:137: Loading the model parameters from flower-recognition-0020.params

[22:26:50] inception_inference.cpp:179: Loading the image ./data/test/0/image_06736.jpg

[22:26:50] inception_inference.cpp:230: Running the forward pass on model to predict the image
[22:26:50] inception_inference.cpp:260: The model predicts the highest accuracy of 7.17001 at index 3
```


## What's next

You can find more ways to run inference and examples here:
1. [Java Inference examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/java/org/apache/mxnetexamples/javaapi/infer)
2. [Scala Inference examples](https://mxnet.incubator.apache.org/tutorials/scala/)
3. [ONNX model inference examples](https://mxnet.incubator.apache.org/tutorials/onnx/inference_on_onnx_model.html)

## References

1. https://github.com/Arsey/keras-transfer-learning-for-oxford102
1. https://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html
2. https://github.com/leleamol/incubator-mxnet/blob/inception-example/cpp-package/example/inference/
3. https://gluon-crash-course.mxnet.io/

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->