from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue
from .proto.tensor_shape_pb2 import TensorShapeProto


def gg(fname):
    import onnx  # 0.2.1
    m = onnx.load(fname)
    nodes_proto = []
    nodes = []
    g = m.graph
    import itertools
    for node in itertools.chain(g.input, g.output):
        nodes_proto.append(node)

    for node in nodes_proto:
        shapeproto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value) for d in node.type.tensor_type.shape.dim])
        nodes.append(NodeDef(
            name=node.name,
            op='Variable',
            input=[],
            attr={
                'dtype': AttrValue(type=node.type.tensor_type.elem_type),
                'shape': AttrValue(shape=shapeproto),
            })
        )

    for node in g.node:
        attr = []
        for s in node.attribute:
            attr.append(' = '.join([str(f[1]) for f in s.ListFields()]))
        attr = ', '.join(attr).encode(encoding='utf_8')

        nodes.append(NodeDef(
            name=node.output[0],
            op=node.op_type,
            input=node.input,
            attr={'parameters': AttrValue(s=attr)},
        ))
    # two pass token replacement, appends opname to object id
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + '_' + node.name

    nodes, mapping = updatenodes(nodes, mapping)
    mapping = smartGrouping(nodes, mapping)
    nodes, mapping = updatenodes(nodes, mapping)

    return GraphDef(node=nodes, versions=VersionDef(producer=22))


def updatenodes(nodes, mapping):
    for node in nodes:
        newname = mapping[node.name]
        node.name = newname
        newinput = []
        for inputnode in list(node.input):
            newinput.append(mapping[inputnode])
            node.input.remove(inputnode)
        node.input.extend(newinput)
    newmap = {}
    for k, v in mapping.items():
        newmap[v] = v
    return nodes, newmap


def findnode(nodes, name):
    """ input: node name
        returns: node object
    """
    for n in nodes:
        if n.name == name:
            return n


def parser(s, nodes, node):
    print(s)
    if len(s) == 0:
        return
    if len(s) > 0:
        if s[0] == node.op:
            print(s[0], node.name, s[1], node.input)
            for n in node.input:
                print(n, s[1])
                parser(s[1], nodes, findnode(nodes, n))
        else:
            return False


# TODO: use recursive parse

def smartGrouping(nodes, mapping):
    # a Fully Conv is: (TODO: check var1.size(0)==var2.size(0))
    # GEMM <-- Variable (c1)
    #  ^-- Transpose (c2) <-- Variable (c3)

    # a Conv with bias is: (TODO: check var1.size(0)==var2.size(0))
    # Add <-- Conv (c2) <-- Variable (c3)
    #  ^-- Variable (c1)
    #
    # gemm = ('Gemm', ('Variable', ('Transpose', ('Variable'))))

    FCcounter = 1
    Convcounter = 1
    for node in nodes:
        if node.op == 'Gemm':
            c1 = c2 = c3 = False
            for name_in in node.input:
                n = findnode(nodes, name_in)
                if n.op == 'Variable':
                    c1 = True
                    c1name = n.name
                if n.op == 'Transpose':
                    c2 = True
                    c2name = n.name
                    if len(n.input) == 1:
                        nn = findnode(nodes, n.input[0])
                        if nn.op == 'Variable':
                            c3 = True
                            c3name = nn.name
                # print(n.op, n.name, c1, c2, c3)
            if c1 and c2 and c3:
                # print(c1name, c2name, c3name)
                mapping[c1name] = 'FC{}/{}'.format(FCcounter, c1name)
                mapping[c2name] = 'FC{}/{}'.format(FCcounter, c2name)
                mapping[c3name] = 'FC{}/{}'.format(FCcounter, c3name)
                mapping[node.name] = 'FC{}/{}'.format(FCcounter, node.name)
                FCcounter += 1
                continue
        if node.op == 'Add':
            c1 = c2 = c3 = False
            for name_in in node.input:
                n = findnode(nodes, name_in)
                if n.op == 'Variable':
                    c1 = True
                    c1name = n.name
                if n.op == 'Conv':
                    c2 = True
                    c2name = n.name
                    if len(n.input) >= 1:
                        for nn_name in n.input:
                            nn = findnode(nodes, nn_name)
                            if nn.op == 'Variable':
                                c3 = True
                                c3name = nn.name

            if c1 and c2 and c3:
                # print(c1name, c2name, c3name)
                mapping[c1name] = 'Conv{}/{}'.format(Convcounter, c1name)
                mapping[c2name] = 'Conv{}/{}'.format(Convcounter, c2name)
                mapping[c3name] = 'Conv{}/{}'.format(Convcounter, c3name)
                mapping[node.name] = 'Conv{}/{}'.format(Convcounter, node.name)
                Convcounter += 1
    return mapping
