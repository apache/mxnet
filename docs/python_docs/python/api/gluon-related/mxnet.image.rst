mxnet.image
===========

.. automodule:: mxnet.image


Image processing functions
--------------------------

.. autosummary::
   :toctree: _autogen

   imdecode
   scale_down
   resize_short
   fixed_crop
   random_crop
   center_crop
   color_normalize
   random_size_crop

Image classifiction
-------------------

Iterators
^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   ImageIter

Augmentations
^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   CreateAugmenter
   Augmenter
   SequentialAug
   RandomOrderAug
   ResizeAug
   ForceResizeAug
   RandomCropAug
   RandomSizedCropAug
   CenterCropAug
   BrightnessJitterAug
   ContrastJitterAug
   SaturationJitterAug
   HueJitterAug
   ColorJitterAug
   LightingAug
   ColorNormalizeAug
   RandomGrayAug
   HorizontalFlipAug
   CastAug

Object detection
----------------

Iterators
^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   ImageDetIter


Augmentations
^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   CreateDetAugmenter
   DetBorrowAug
   DetRandomSelectAug
   DetHorizontalFlipAug
   DetRandomCropAug
   DetRandomPadAug
