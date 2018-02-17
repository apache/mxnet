# Real-time Object Detection with MXNet On The Raspberry Pi  

This tutorial shows developers who work with the Raspberry Pi or similar embedded ARM-based devices how to compile MXNet for those devices and run a pretrained deep network model. It also shows how to use AWS IoT to manage and monitor MXNet models running on your devices.

## What's In This Tutorial?

This tutorial shows how to:

1. Use MXNet to set up a real-time object classifier on a Raspberry Pi 3 device.

2. Connect the local Raspberry Pi model to the AWS Cloud with AWS IoT to get real-time updates from the device.

### Who's This Tutorial For?

It assumes that you are familiar with the Raspbian operating system and the [Raspberry Pi ecosystem](https://www.raspberrypi.org/) and are somewhat familiar with machine learning, MXNet, and [AWS IoT](https://aws.amazon.com/iot/). All code is written in Python 2.7.

### How to Use This Tutorial

To follow this tutorial, you must set up your Pi as instructed (preferably from a fresh Raspbian install), and then create the files and run the bash commands described below. All instructions described are can be executed on the Raspberry Pi directly or via SSH.

You will accomplish the following:

- Build and Install MXNet with Python bindings on your Raspbian Based Raspberry Pi
- Fetch and run a pre-trained MXNet model on your Pi
- Create a real-time video analysis application for the Pi
- Connect the application to the AWS IoT service

## Prerequisites

To complete this tutorial, you need:

* Raspbian Wheezy or later, which can be downloaded [here](https://www.raspberrypi.org/downloads/raspbian/), loaded onto a 8GB+ micro SD card (with at least 4GB+ free)
* A [Raspberry Pi 3](https://www.raspberrypi.org/blog/raspberry-pi-3-on-sale/) or equivalent Raspberry Pi with 1GB+ of RAM
* A [Raspberry Pi Camera Module](https://www.raspberrypi.org/products/camera-module/) [activated and running with the corresponding Python module](http://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/) (for the real-time video analysis with the deep network model)
* An AWS account With AWS IoT enabled and the [AWS IoT Python SDK](https://github.com/aws/aws-iot-device-sdk-python) (for remote, real-time managing and monitoring of the model running on the Pi)
* The [cv2 Python library](http://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/) for the Pi

## Building MXNet for The Pi

The first step is to get MXNet with the Python bindings running on your Raspberry Pi 3. There is a tutorial for that provided [here](http://mxnet.io/install/index.html). The linked tutorial walks you through downloading the dependencies, and building the full MXNet library for the Pi with the ARM specific compile flags. Be sure to build the library with open CV as we will be using a model that requires it to process images. Then you will register the Python bindings to MXNet. After this is done you should test that your installation works by opening a python REPL on your Pi and typing the following commands:


```bash
python
>>> import mxnet as mx
```

*Note: If you are getting memory allocation failed errors at this point (or at any point in this tutorial) it is likely because the full MXNet library takes up a large amount of RAM when loaded. You might want to kill the GUI and other processes that are occupying memory.*


## Running A Pre-Trained Inception Model on The Pi

We are now ready to load a pre-trained model and run inference on the Pi. We will be using a simple object recognition model trained on the ImageNet data set. The model is called batch normalized Inception network (or Inception_BN for short) and it is found in the MXNet model zoo.

### Getting the Model

The first step is to download, unzip, and set up the pre-trained deep network model files that we will be using to classify images. To do this run the following commands in your home directory:

```bash
curl --header 'Host: data.mxnet.io' --header 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:45.0) Gecko/20100101 Firefox/45.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --header 'Referer: http://data.mxnet.io/models/imagenet/' --header 'Connection: keep-alive' 'http://data.mxnet.io/models/imagenet/inception-bn.tar.gz' -o 'inception-bn.tar.gz' -L

tar -xvzf inception-bn.tar.gz

mv Inception_BN-0039.params Inception_BN-0000.params
```

### Running the Model

The next step is to create a python script to load the model, and run inference on local image files. To do this create a new file in your home directory called inception_predict.py and add the following code to it:


```python
# inception_predict.py

import mxnet as mx
import numpy as np
import cv2, os, urllib
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# Load the symbols for the networks
with open('synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

# Load the network parameters
sym, arg_params, aux_params = mx.model.load_checkpoint('Inception_BN', 0)

# Load the network into an MXNet module and bind the corresponding parameters
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)
 
'''
Function to predict objects by giving the model a pointer to an image file and running a forward pass through the model.

inputs:
filename = jpeg file of image to classify objects in
mod = the module object representing the loaded model
synsets = the list of symbols representing the model
N = Optional parameter denoting how many predictions to return (default is top 5)

outputs:
python list of top N predicted objects and corresponding probabilities
'''
def predict(filename, mod, synsets, N=5):
    tic = time.time()
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    print "pre-processed image in "+str(time.time()-tic)
 
    toc = time.time()
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    print "forward pass in "+str(time.time()-toc)
 
 
    topN = []
    a = np.argsort(prob)[::-1]
    for i in a[0:N]:
        print('probability=%f, class=%s' %(prob[i], synsets[i]))
        topN.append((prob[i], synsets[i]))
    return topN


# Code to download an image from the internet and run a prediction on it
def predict_from_url(url, N=5):
    filename = url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print "Failed to download"
    else:
        return predict(filename, mod, synsets, N)

# Code to predict on a local file
def predict_from_local_file(filename, N=5):        
    return predict(filename, mod, synsets, N)
```

Now that we have defined inception_predict.py we can test that the model is running correctly. Open a Python REPL in your home directory and enter the following:

```bash
python
>>> import inception_predict
>>> predict_from_url("http://imgur.com/HzafyBA")
```

This should give a reasonable prediction for the fluffy cow in this [image](http://imgur.com/HzafyBA). 


## Running an Inception on Real-Time Video From PiCamera

We can now move on to using this network for object detection in real-time video from the PiCamera.

Doing this requires sending the images that the camera is capturing to the prediction code that we created in the previous step. To do this, create a new file in your home directory called camera_test.py and add the following code to it:


```python
# camera_test.py

import picamera
import inception_predict

# Create camera interface
camera = picamera.PiCamera()
while True:
    # Take the jpg image from camera
    print "Capturing"
    filename = '/home/pi/cap.jpg'
    # Show quick preview of what's being captured
    camera.start_preview()
    camera.capture(filename)
    camera.stop_preview()
    
    # Run inception prediction on image
    print "Predicting"
    topn = inception_predict.predict_from_local_file(filename, N=5)
    
    # Print the top N most likely objects in image (default set to 5, change this in the function call above)
    print topn
```

You can then run this file by entering the following command:

```bash
python camera_test.py
```

If camera_test.py is working you should see a preview every few seconds of the image that is being captured and fed to the model, as well as predicted classes for objects in the image being written to the terminal. 

Try pointing the PiCamera at a few different objects and see what predictions the network comes out with.

## Connecting Our Model To The AWS Cloud

We can now move on to adding the code to send the predictions that this real-time model is making locally to the AWS cloud if certain conditions are met.

The first step is to set up an AWS account if you don't have one yet. Then go to the [AWS IoT dashboard](https://us-west-2.console.aws.amazon.com/iotv2/home?region=us-west-2#/thinghub) and register a new device.

After the device is registered, download and copy the corresponding rootCA, Certificate, and Private key to your home directory. Note the unique endpoint of your device shadow on the AWS IoT Dashboard.

We will now build an application, based off the code in camera_test.py, which will send a message to the cloud whenever a wine bottle is detected in a frame by the PiCamera.

To do this create a new file in your home directory called wine_alerter.py and add the following code to it:


```python
# wine_alerter.py

import AWSIoTPythonSDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import sys
import logging
import time
import getopt
import picamera
import inception_predict

# Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

# Usage
usageInfo = """Usage:
 
Use certificate based mutual authentication:
python wine_alerter.py -e <endpoint> -r <rootCAFilePath> -c <certFilePath> -k <privateKeyFilePath>
 
Use MQTT over WebSocket:
python wine_alerter.py -e <endpoint> -r <rootCAFilePath> -w
 
Type "python wine_alerter.py -h" for available options.
"""

# Help info
helpInfo = """-e, --endpoint
    Your AWS IoT custom endpoint
-r, --rootCA
    Root CA file path
-c, --cert
    Certificate file path
-k, --key
    Private key file path
-w, --websocket
    Use MQTT over WebSocket
-h, --help
    Help information
"""
 
# Read in command-line parameters
useWebsocket = False
host = ""
rootCAPath = ""
certificatePath = ""
privateKeyPath = ""
try:
    opts, args = getopt.getopt(sys.argv[1:], "hwe:k:c:r:", ["help", "endpoint=", "key=","cert=","rootCA=", "websocket"])
    if len(opts) == 0:
        raise getopt.GetoptError("No input parameters!")
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(helpInfo)
            exit(0)
        if opt in ("-e", "--endpoint"):
            host = arg
        if opt in ("-r", "--rootCA"):
            rootCAPath = arg
        if opt in ("-c", "--cert"):
            certificatePath = arg
        if opt in ("-k", "--key"):
            privateKeyPath = arg
        if opt in ("-w", "--websocket"):
            useWebsocket = True
except getopt.GetoptError:
    print(usageInfo)
    exit(1)

# Missing configuration notification
missingConfiguration = False
if not host:
    print("Missing '-e' or '--endpoint'")
    missingConfiguration = True
if not rootCAPath:
    print("Missing '-r' or '--rootCA'")
    missingConfiguration = True
if not useWebsocket:
    if not certificatePath:
        print("Missing '-c' or '--cert'")
        missingConfiguration = True
    if not privateKeyPath:
        print("Missing '-k' or '--key'")
        missingConfiguration = True
if missingConfiguration:
    exit(2)


# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)


# Init AWSIoTMQTTClient For Publish/Subscribe Communication With Server
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient("basicPubSub", useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, 443)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient("basicPubSub")
    myAWSIoTMQTTClient.configureEndpoint(host, 8883)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)


# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec


# Connect and subscribe to AWS IoT
myAWSIoTMQTTClient.connect()
myAWSIoTMQTTClient.subscribe("sdk/test/Python", 1, customCallback)
time.sleep(2)


# Start the Camera and tell the Server we are alive
print "Running camera"
myAWSIoTMQTTClient.publish("sdk/test/Python", "New Message: Starting Camera", 0)
camera = picamera.PiCamera()

# Capture forever (this is a modified version of camera_test.py)
while True:
    filename = '/home/pi/cap.jpg'
    camera.start_preview()
    camera.capture(filename)
    camera.stop_preview()
    topn = inception_predict.predict_from_local_file(filename, N=5)
    
    # Check if either of the top two predictions are wine related and publish a message if it is
    # you can change 'wine' here to anything you want to alert the server about detecting
    if 'wine' in topn[0][1] or 'wine' in topn[1][1]: 
        myAWSIoTMQTTClient.publish("sdk/test/Python", "New Message: WINE DETECTED!", 0)
```

You can then run this file by entering the following command

```bash
python wine_alerter.py -e <endpointURL> -r <rootCAFilePath> -c <certFilePath> -k <privateKeyFilePath>
```

If this is working you should see the same kind of image preview you did with camera_test.py every few seconds, however the console will only print a message now when a wine bottle is detected in the shot (you can edit the bottom lines in the wine_alerter.py code to make this alert for any object label from the [ImageNet-11k dataset](http://image-net.org/index) that you specify).

You can open up the activity tab for the thing that you registered on the AWS IoT Dashboard and see the corresponding messages pushed to the server whenever a wine bottle is detected in a camera shot. Even if network connectivity periodically fails, AWS IoT will push updates out to the server when possible, allowing this system to reliably let you know when there is wine around.

## Summary
You now have a Raspberry Pi running a pre-trained MXNet model fully locally. This model is also linked to the cloud via AWS IoT and can reliably alert AWS whenever it sees a wine bottle.

You can now extend this demo to create more interesting applications, such as using AWS IoT to push model updates to your Pi, loading different pre-trained models from the MXNet Model Zoo onto the Pi, or even training full end-to-end models on the Pi.
