
Executor
========




.. class:: Executor

   An executor is a realization of a symbolic architecture defined by a :class:`Symbol`.
   The actual forward and backward computation specified by the network architecture can
   be carried out with an executor.




.. function:: bind(sym, ctx, args; args_grad=Dict(), aux_states=Dict(), grad_req=GRAD_WRITE)

   Create an :class:`Executor` by binding a :class:`Symbol` to concrete :class:`NDArray`.

   :param Symbol sym: the network architecture describing the computation graph.
   :param Context ctx: the context on which the computation should run.
   :param args: either a list of :class:`NDArray` or a dictionary of name-array pairs. Concrete
          arrays for all the inputs in the network architecture. The inputs typically include
          network parameters (weights, bias, filters, etc.), data and labels. See :func:`list_arguments`
          and :func:`infer_shape`.
   :param args_grad: TODO
   :param aux_states:
   :param grad_req:



