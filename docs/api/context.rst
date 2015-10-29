
Context
=======




.. class:: Context

   A context describes the device type and id on which computation should be carried on.




.. function:: cpu(dev_id=0)

   :param Int dev_id: the CPU id.

   Get a CPU context with a specific id. ``cpu()`` is usually the default context for many
   operations when no context is specified.




.. function:: gpu(dev_id=0)

   :param Int dev_id: the GPU device id.

   Get a GPU context with a specific id. The K GPUs on a node is typically numbered as 0,...,K-1.



