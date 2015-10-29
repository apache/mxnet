
Data Providers
==============

Data providers are wrappers that load external data, be it images, text, or general tensors,
and split it into mini-batches so that the model can consume the data in a uniformed way.




.. class:: AbstractDataProvider

   The root type for all data provider. A data provider should implement the following interfaces:

   .. function:: get_batch_size(provider) -> Int

      :param AbstractDataProvider provider: the data provider.
      :return: the mini-batch size of the provided data. All the provided data should have the
               same mini-batch size (i.e. the last dimension).

   .. function:: provide_data(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

      :param AbstractDataProvider provider: the data provider.
      :return: a vector of (name, shape) pairs describing the names of the data it provides, and
               the corresponding shapes.

   .. function:: provide_label(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

      :param AbstractDataProvider provider: the data provider.
      :return: a vector of (name, shape) pairs describing the names of the labels it provides, and
               the corresponding shapes.

   The difference between *data* and *label* is that during
   training stage, both *data* and *label* will be feeded into the model, while during
   prediction stage, only *data* is loaded. Otherwise, they could be anything, with any names, and
   of any shapes. The provided data and label names here should match the input names in a target
   :class:`Symbol`.

   A data provider should also implement the Julia iteration interface, in order to allow iterating
   through the data set. The provider will be called in the following way:

   .. code-block:: julia

      for batch in provider
        data = get_data(provider, batch)
      end

   which will be translated by Julia compiler into

   .. code-block:: julia

      state = Base.start(provider)
      while !Base.done(provider, state)
        (batch, state) = Base.next(provider, state)
        data = get_data(provider, batch)
      end

   The detailed interface function is listed below:

   .. function:: Base.eltype(provider) -> AbstractDataBatch

      :param AbstractDataProvider provider: the data provider.
      :return: the specific subtype representing a data batch. See :class:`AbstractDataBatch`.

   .. function:: Base.start(provider) -> AbstractDataProviderState

      :param AbstractDataProvider provider: the data provider.

      This function is always called before iterating into the dataset. It should initialize
      the iterator, reset the index, and do data shuffling if needed.

   .. function:: Base.done(provider, state) -> Bool

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataProviderState state: the state returned by :func:`Base.start` :func:`Base.next`.
      :return: true if there is no more data to iterate in this dataset.

   .. function:: Base.next(provider) -> (AbstractDataBatch, AbstractDataProviderState)

      :param AbstractDataProvider provider: the data provider.
      :return: the current data batch, and the state for the next iteration.

   Note sometimes you are wrapping an existing data iterator (e.g. the built-in libmxnet data iterator) that
   is built with a different convention. It might be difficult to adapt to the interfaces stated here. In this
   case, you can safely assume that

   * :func:`Base.start` will always be called, and called only once before the iteration starts.
   * :func:`Base.done` will always be called at the beginning of every iteration and always be called once.
   * If :func:`Base.done` return true, the iteration will stop, until the next round, again, starting with
     a call to :func:`Base.start`.
   * :func:`Base.next` will always be called only once in each iteration. It will always be called after
     one and only one call to :func:`Base.done`; but if :func:`Base.done` returns true, :func:`Base.next` will
     not be called.

   With those assumptions, it will be relatively easy to adapt any existing iterator. See the implementation
   of the built-in :class:`MXDataProvider` for example.




.. class:: AbstractDataProviderState

   Base type for data provider states.




.. class:: AbstractDataBatch

   Base type for a data mini-batch. It should implement the following interfaces:

   .. function:: count_samples(provider, batch) -> Int

      :param AbstractDataBatch batch: the data batch object.
      :return: the number of samples in this batch. This number should be greater than 0, but
               less than or equal to the batch size. This is used to indicate at the end of
               the data set, there might not be enough samples for a whole mini-batch.

   .. function:: get_data(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of data in this batch, should be in the same order as declared in
               :func:`provide_data() <AbstractDataProvider.provide_data>`.

               The last dimension of each :class:`NDArray` should always match the batch_size, even when
               :func:`count_samples` returns a value less than the batch size. In this case,
               the data provider is free to pad the remaining contents with any value.

   .. function:: get_label(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of labels in this batch. Similar to :func:`get_data`.


   The following utility functions will be automatically defined.

   .. function:: get(provider, batch, name) -> NDArray

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param Base.Symbol name: the name of the data to get, should be one of the names
             provided in either :func:`provide_data() <AbstractDataProvider.provide_data>`
             or :func:`provide_label() <AbstractDataprovider.provide_label>`.
      :return: the corresponding data array corresponding to that name.

   .. function:: load_data!(provider, batch, targets)

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param targets: the targets to load data into.
      :type targets: Vector{Vector{SlicedNDArray}}

      The targets is a list of the same length as number of data provided by this provider.
      Each element in the list is a ``Vector{SlicedNDArray}``. This vector described a
      spliting of this data batch into different slices, each slice is specified by
      a slice-ndarray pair, where *slice* specify the range of samples in the mini-batch
      that should be loaded into the corresponding *ndarray*.

      This utility function is used in data parallelization, where a mini-batch is splited
      and computed on several different devices.

   .. function:: load_label!(provider, batch, targets)

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param targets: the targets to load label into.
      :type targets: Vector{Vector{SlicedNDArray}}

      The same as :func:`load_data!`, except that this is for loading labels.




.. class:: SlicedNDArray

   A alias type of ``Pair{UnitRange{Int},NDArray}``.




.. class:: MXDataProvider

   A data provider that wrap built-in data iterators from libmxnet.




Built-in data providers in libmxnet
-----------------------------------

.. function:: ImageRecordIter(...)

   Can also be called with the alias ``ImageRecordProvider``.
   Create iterator for dataset packed in recordio.
   
   :param Base.Symbol data_name: keyword argument, default ``:data``. The name of the data.
   :param Base.Symbol label_name: keyword argument, default ``:softmax_label``. The name of the label. Could be ``nothing`` if no label is presented in this dataset.
   
   :param path_imglist: Dataset Param: Path to image list.
   :type path_imglist: string, optional, default=''
   
   
   :param path_imgrec: Dataset Param: Path to image record file.
   :type path_imgrec: string, optional, default='./data/imgrec.rec'
   
   
   :param label_width: Dataset Param: How many labels for an image.
   :type label_width: int, optional, default='1'
   
   
   :param data_shape: Dataset Param: Shape of each instance generated by the DataIter.
   :type data_shape: Shape(tuple), required
   
   
   :param preprocess_threads: Backend Param: Number of thread to do preprocessing.
   :type preprocess_threads: int, optional, default='4'
   
   
   :param verbose: Auxiliary Param: Whether to output parser information.
   :type verbose: boolean, optional, default=True
   
   
   :param num_parts: partition the data into multiple parts
   :type num_parts: int, optional, default='1'
   
   
   :param part_index: the index of the part will read
   :type part_index: int, optional, default='0'
   
   
   :param shuffle: Augmentation Param: Whether to shuffle data.
   :type shuffle: boolean, optional, default=False
   
   
   :param seed: Augmentation Param: Random Seed.
   :type seed: int, optional, default='0'
   
   
   :param batch_size: Batch Param: Batch size.
   :type batch_size: int (non-negative), required
   
   
   :param round_batch: Batch Param: Use round robin to handle overflow batch.
   :type round_batch: boolean, optional, default=True
   
   
   :param prefetch_buffer: Backend Param: Number of prefetched parameters
   :type prefetch_buffer: , optional, default=4
   
   
   :param rand_crop: Augmentation Param: Whether to random crop on the image
   :type rand_crop: boolean, optional, default=False
   
   
   :param crop_y_start: Augmentation Param: Where to nonrandom crop on y.
   :type crop_y_start: int, optional, default='-1'
   
   
   :param crop_x_start: Augmentation Param: Where to nonrandom crop on x.
   :type crop_x_start: int, optional, default='-1'
   
   
   :param max_rotate_angle: Augmentation Param: rotated randomly in [-max_rotate_angle, max_rotate_angle].
   :type max_rotate_angle: int, optional, default='0'
   
   
   :param max_aspect_ratio: Augmentation Param: denotes the max ratio of random aspect ratio augmentation.
   :type max_aspect_ratio: float, optional, default=0
   
   
   :param max_shear_ratio: Augmentation Param: denotes the max random shearing ratio.
   :type max_shear_ratio: float, optional, default=0
   
   
   :param max_crop_size: Augmentation Param: Maximum crop size.
   :type max_crop_size: int, optional, default='-1'
   
   
   :param min_crop_size: Augmentation Param: Minimum crop size.
   :type min_crop_size: int, optional, default='-1'
   
   
   :param max_random_scale: Augmentation Param: Maxmum scale ratio.
   :type max_random_scale: float, optional, default=1
   
   
   :param min_random_scale: Augmentation Param: Minimum scale ratio.
   :type min_random_scale: float, optional, default=1
   
   
   :param max_img_size: Augmentation Param: Maxmum image size after resizing.
   :type max_img_size: float, optional, default=1e+10
   
   
   :param min_img_size: Augmentation Param: Minimum image size after resizing.
   :type min_img_size: float, optional, default=0
   
   
   :param rotate: Augmentation Param: Rotate angle.
   :type rotate: int, optional, default='-1'
   
   
   :param fill_value: Augmentation Param: Maximum value of illumination variation.
   :type fill_value: int, optional, default='255'
   
   
   :param mirror: Augmentation Param: Whether to mirror the image.
   :type mirror: boolean, optional, default=False
   
   
   :param rand_mirror: Augmentation Param: Whether to mirror the image randomly.
   :type rand_mirror: boolean, optional, default=False
   
   
   :param mean_img: Augmentation Param: Mean Image to be subtracted.
   :type mean_img: string, optional, default=''
   
   
   :param mean_r: Augmentation Param: Mean value on R channel.
   :type mean_r: float, optional, default=0
   
   
   :param mean_g: Augmentation: Mean value on G channel.
   :type mean_g: float, optional, default=0
   
   
   :param mean_b: Augmentation: Mean value on B channel.
   :type mean_b: float, optional, default=0
   
   
   :param scale: Augmentation Param: Scale in color space.
   :type scale: float, optional, default=1
   
   
   :param max_random_contrast: Augmentation Param: Maximum ratio of contrast variation.
   :type max_random_contrast: float, optional, default=0
   
   
   :param max_random_illumination: Augmentation Param: Maximum value of illumination variation.
   :type max_random_illumination: float, optional, default=0
   
   :return: the constructed :class:`MXDataProvider`.



.. function:: MNISTIter(...)

   Can also be called with the alias ``MNISTProvider``.
   Create iterator for MNIST hand-written digit number recognition dataset.
   
   :param Base.Symbol data_name: keyword argument, default ``:data``. The name of the data.
   :param Base.Symbol label_name: keyword argument, default ``:softmax_label``. The name of the label. Could be ``nothing`` if no label is presented in this dataset.
   
   :param image: Dataset Param: Mnist image path.
   :type image: string, optional, default='./train-images-idx3-ubyte'
   
   
   :param label: Dataset Param: Mnist label path.
   :type label: string, optional, default='./train-labels-idx1-ubyte'
   
   
   :param batch_size: Batch Param: Batch Size.
   :type batch_size: int, optional, default='128'
   
   
   :param shuffle: Augmentation Param: Whether to shuffle data.
   :type shuffle: boolean, optional, default=True
   
   
   :param flat: Augmentation Param: Whether to flat the data into 1D.
   :type flat: boolean, optional, default=False
   
   
   :param seed: Augmentation Param: Random Seed.
   :type seed: int, optional, default='0'
   
   
   :param silent: Auxiliary Param: Whether to print out data info.
   :type silent: boolean, optional, default=False
   
   
   :param num_parts: partition the data into multiple parts
   :type num_parts: int, optional, default='1'
   
   
   :param part_index: the index of the part will read
   :type part_index: int, optional, default='0'
   
   
   :param prefetch_buffer: Backend Param: Number of prefetched parameters
   :type prefetch_buffer: , optional, default=4
   
   :return: the constructed :class:`MXDataProvider`.






