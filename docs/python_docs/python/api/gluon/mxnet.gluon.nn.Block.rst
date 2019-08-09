Block
=====

.. currentmodule:: mxnet.gluon.nn

.. autoclass:: Block
   :members:
   :inherited-members:

   ..
      .. automethod:: __init__


   .. rubric:: Handle model parameters:

   .. autosummary::
      :toctree: _autogen

      Block.initialize
      Block.save_parameters
      Block.load_parameters
      Block.collect_params
      Block.cast
      Block.apply

   .. rubric:: Run computation

   .. autosummary::
      :toctree: _autogen

      Block.forward

   .. rubric:: Debugging

   .. autosummary::
      :toctree: _autogen

      Block.summary

   .. rubric:: Advanced API for customization


   .. autosummary::
      :toctree: _autogen

      Block.name_scope
      Block.register_child
      Block.register_forward_hook
      Block.register_forward_pre_hook

   .. rubric:: Attributes

   .. autosummary::

      Block.name
      Block.params
      Block.prefix


   .. warning::

      The following two APIs are deprecated since `v1.2.1
      <https://github.com/apache/incubator-mxnet/releases/tag/1.2.1>`_.

      .. autosummary::
          :toctree: _autogen

          Block.save_params
          Block.load_params
