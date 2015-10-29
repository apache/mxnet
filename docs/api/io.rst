
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

   .. function:: count_samples(batch) -> Int

      :param AbstractDataBatch batch: the data batch object.
      :return: the number of samples in this batch. This number should be greater than 0, but
               less than or equal to the batch size. This is used to indicate at the end of
               the data set, there might not be enough samples for a whole mini-batch.

   .. function:: get_data(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of data in this batch, should be in the same order as declared in
               :func:`provide_data() <AbstractDataProvider.provide_data>`. The last dimension
               of each :class:`NDArray` should match the value returned by :func:`count_samples`.

   .. function:: get_label(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of labels in this batch. Similar to :func:`get_data`.



