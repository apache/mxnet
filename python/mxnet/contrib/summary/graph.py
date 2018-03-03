from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue


def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')  # singlequote will be escaped by tensorboard
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append({'name': replace(uname, scope), 'op': n.kind(), 'inputs': inputs, 'attr': attrs})

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append({'name': replace(uname, scope), 'op': 'Parameter', 'inputs': [], 'attr': str(n.type())})

    return nodes


def graph(model, args, verbose=False):
    import torch
    with torch.onnx.set_training(model, False):
        trace, _ = torch.jit.trace(model, args)
    torch.onnx._optimize_trace(trace, False)
    graph = trace.graph()
    if verbose:
        print(graph)
    list_of_nodes = parse(graph)
    nodes = []
    for node in list_of_nodes:
        nodes.append(
            NodeDef(name=node['name'], op=node['op'], input=node['inputs'],
                    attr={'lanpa': AttrValue(s=node['attr'].encode(encoding='utf_8'))}))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
