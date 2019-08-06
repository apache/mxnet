Gluon
=====

Getting started
---------------

.. container:: cards

   .. card::
      :title: A 60-minute Gluon crash course
      :link: ../../crash-course/index.html

      Six 10-minute tutorials covering the core concepts of MXNet using the Gluon API.

   .. card::
      :title: Gluon - Neural network building blocks
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/gluon.html

      An introduction to defining and training neural networks with Gluon.

   .. card::
      :title: Gluon: from experiment to deployment
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/gluon_from_experiment_to_deployment.html

      An end to end tutorial on working with the MXNet Gluon API.

   .. card::
      :title: Custom Layers for Beginners
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/custom_layer.html

      A guide to implementing custom layers for beginners.

   .. card::
      :title: Logistic regression using Gluon API explained
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/logistic_regression_explained.html

      Implementing logistic regression using the Gluon API.

   .. card::
      :title: Saving and Loading Gluon Models
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html

      Saving and loading trained models.

   .. card::
      :title: Using pre-trained models in MXNet
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/pretrained_models.html

      Using pre-trained models with Apache MXNet.

Data
----

.. container:: cards

   .. card::
      :title: Data Loading
      :link: data.html

      How to load data for training.

   .. card::
      :title: Image Augmentation
      :link: image-augmentation.html

      Boost your training dataset with image augmentation.

   .. card::
      :title: Data Augmentation
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/data_augmentation.html

      A guide to data augmentation.

   .. card::
      :title: Gluon Datasets and DataLoader
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/datasets.html

      A guide to loading data using the Gluon API.

   .. card::
      :title: NDArray - Scientific computing on CPU and GPU
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/ndarray.html

      A guide to the NDArray data structure.

Training
--------

.. container:: cards

   .. card::
      :title: Neural Networks
      :link: nn.html

      How to use Layers and Blocks.

   .. card::
      :title: Normalization Blocks
      :link: normalization/normalization.html

      Understand usage of normalization layers (such as BatchNorm).

   .. card::
      :title: Activation Blocks
      :link: activations/activations.html

      Understand usage of activation layers (such as ReLU).

   .. card::
      :title: Loss Functions
      :link: loss.html

      How to use loss functions for predicting outputs.

   .. card::
      :title: Initializing Parameters
      :link: init.html

      How to use the init function.

   .. card::
      :title: Parameter Management
      :link: parameters.html

      How to manage parameters.

   .. card::
      :title: Learning Rate Finder
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_finder.html

      How to use the Learning Rate Finder to find a good learning rate.

   .. card::
      :title: Learning Rate Schedules
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules.html

      How to schedule Learning Rate change over time.

   ..
      .. card::
         :title: Optimizer
         :link: optimizer.html

         How to use optimizer.
   ..

   .. card::
      :title: Trainer
      :link: trainer.html

      How to update neural network parameters using an optimization method.

   .. card::
      :title: Autograd API
      :link: ../autograd/autograd.html

      How to use Automatic Differentiation with the Autograd API.

Advanced Topics
---------------

.. container:: cards

   .. card::
      :title: Naming
      :link: naming.html

      Best practices for the naming of things.

   .. card::
      :title: Custom Layers
      :link: custom-layer.html

      A guide to implementing custom layers.

   .. card::
      :title: Custom Operators
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/customop.html

      Building custom operators with numpy.

   .. card::
      :title: Custom Loss
      :link: custom-loss/custom-loss.html

      A guide to implementing custom losses.

   .. card::
      :title: Gotchas using NumPy in Apache MXNet
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/gotchas_numpy_in_mxnet.html

      Common misconceptions when using NumPy in Apache MXNet.

   .. card::
      :title: Hybrid- Faster training and easy deployment
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/hybrid.html

      Combines declarative and imperative programming using HybridBlock.

   .. card::
      :title: Hybridize
      :link: hybridize.html

      Speed up training with hybrid networks.

   .. card::
      :title: Advanced Learning Rate Schedules
      :link: https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/learning_rate_schedules_advanced.html

      Advanced exploration of Learning Rate shapes.


Applications Topics
-------------------

.. container:: cards

   .. card::
      :title: Image Tutorials
      :link: image/index.html

      How to create deep learning models for images.

   .. card::
      :title: Text Tutorials
      :link: text/index.html

      How to create deep learning models for text.


.. toctree::
   :hidden:
   :maxdepth: 1

   ../../crash-course
   custom_layer_beginners
   data
   image-augmentation
   data_augmentation
   nn
   normalization/normalization
   activations/activations
   loss
   custom-loss/custom-loss
   init
   parameters
   trainer
   naming
   custom-layer
   customop
   hybridize
   image/index
   text/index
