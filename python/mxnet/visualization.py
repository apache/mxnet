from .symbol import Symbol
import json
import re
import copy

def _str2tuple(string):
    return re.findall("\d+", string)

def network2dot(title, symbol, shape=None):
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0]) # TODO(xxx): check careful
    node_attr = {"shape":"box", "fixedsize":"true", "width":"1.3", "height":"0.8034", "style":"filled"}
    dot = Digraph(name=title)
    # make nodes
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = "%s_%d" % (op, i)
        # input data
        if i in heads and op == "null":
            label=node["name"]
            attr = copy.deepcopy(node_attr)
            dot.node(name=name, label=label, **attr)
        if op == "null":
            continue
        elif op == "Convolution":
            label="Convolution\n%sx%s/%s, %s" % (_str2tuple(node["param"]["kernel"])[0],
                                                 _str2tuple(node["param"]["kernel"])[1],
                                                 _str2tuple(node["param"]["stride"])[0],
                                                 node["param"]["num_filter"])
            attr = copy.deepcopy(node_attr)
            attr["color"] = "royalblue1"
            dot.node(name=name, label=label, **attr)
        elif op == "FullyConnected":
            label="FullyConnected\n%s" % node["param"]["num_hidden"]
            attr = copy.deepcopy(node_attr)
            attr["color"] = "royalblue1"
            dot.node(name=name, label=label, **attr)
        elif op == "BatchNorm":
            label = "BatchNorm"
            attr = copy.deepcopy(node_attr)
            attr["color"] = "orchid1"
            dot.node(name=name, label=label, **attr)
        elif op == "Concat":
            label = "Concat"
            attr = copy.deepcopy(node_attr)
            attr["color"] = "seagreen1"
            dot.node(name=name, label=label, **attr)
        elif op == "Flatten":
            label = "Flatten"
            attr = copy.deepcopy(node_attr)
            attr["color"] = "seagreen1"
            dot.node(name=name, label=label, **attr)
        elif op == "Reshape":
            label = "Reshape"
            attr = copy.deepcopy(node_attr)
            attr["color"] = "seagreen1"
            dot.node(name=name, label=label, **attr)
        elif op == "Pooling":
            label = "Pooling\n%s, %sx%s/%s" % (node["param"]["pool_type"],
                                               _str2tuple(node["param"]["kernel"])[0],
                                               _str2tuple(node["param"]["kernel"])[1],
                                               _str2tuple(node["param"]["stride"])[0])
            attr = copy.deepcopy(node_attr)
            attr["color"] = "firebrick2"
            dot.node(name=name, label=label, **attr)
        elif op == "Activation" or op == "LeakyReLU":
            label = "%s\n%s" % (op, node["param"]["act_type"])
            attr = copy.deepcopy(node_attr)
            attr["color"] = "salmon"
            dot.node(name=name, label=label, **attr)
        else:
            label = op
            attr = copy.deepcopy(node_attr)
            attr["color"] = "olivedrab1"
            dot.node(name=name, label=label, **attr)

    # add edges
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = "%s_%d" % (op, i)
        if op == "null":
            continue
        else:
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = "%s_%d" % (input_node["op"], item[0])
                if input_node["op"] != "null" or item[0] in heads:
                    # add shape into label
                    attr = {"dir":"back"}
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot




