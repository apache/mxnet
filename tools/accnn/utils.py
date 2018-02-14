# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import copy
import json
import ast


def load_model(args):
  devs = mx.cpu() if args.gpus == None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
  return mx.model.FeedForward.load(args.model, args.load_epoch, ctx=devs)

def topsort(nodes):
  n = len(nodes)
  deg = [0]*n
  g = [[] for _ in xrange(n)]
  for i,node in enumerate(nodes):
    if 'inputs' in node:
      for j in node['inputs']:
        deg[i] += 1
        g[j[0]].append(i)
  from collections import deque
  q = deque([i for i in xrange(n) if deg[i]==0])
  res = []
  for its in xrange(n):
    i = q.popleft()
    res.append(nodes[i])
    for j in g[i]:
      deg[j] -= 1
      if deg[j] == 0:
        q.append(j)
  new_ids=dict([(node['name'],i) for i,node in enumerate(res)])
  for node in res:
    if 'inputs' in node:
      for j in node['inputs']:
        j[0]=new_ids[nodes[j[0]]['name']]
  return res

def is_input(node):
  name = node['name']
  return len(node['inputs']) == 0 and ('weight' not in name) and ('bias' not in name) and ('label' not in name)

def sym_factory(node, data):
  name = node['name']
  params = {}
  if 'param' in node:
    for k, v in node['param'].items():
      try:
        params[k] = ast.literal_eval(v)
      except ValueError, e:
        params[k] = v
  return getattr(mx.symbol, node['op'])(data=data, name=name, **params)

def replace_conv_layer(layer_name, old_model, sym_handle, arg_handle):
  conf = json.loads(old_model.symbol.tojson())
  sym_dict = {}
  nodes = conf['nodes']
  nodes = topsort(nodes)
  res_sym = None
  new_model = old_model
  for i,node in enumerate(nodes):
    sym = None
    if is_input(node):
      sym = mx.symbol.Variable(name='data')
    elif node['op'] != 'null':
      input_nodes = [nodes[int(j[0])] for j in node['inputs']]
      datas = [input_node['name'] for input_node in input_nodes\
                                  if not input_node['name'].startswith(node['name'])]
      try:
        data=sym_dict[datas[0]]
      except Exception, e:
        print 'can not find symbol %s'%(datas[0])
        raise e
      if node['name'] == layer_name:
        sym = sym_handle(data, node)
      else:
        sym = sym_factory(node, data)
    if sym:
      sym_dict[node['name']] = sym
      res_sym = sym

  arg_params = copy.deepcopy(old_model.arg_params)
  if layer_name:
    arg_shapes, _, _ = res_sym.infer_shape(data=(1,3,224,224))
    arg_names = res_sym.list_arguments()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    try:
      arg_handle(arg_shape_dic, arg_params)
    except Exception, e:
      raise Exception('Exception in arg_handle')

  new_model = mx.model.FeedForward(
                symbol=res_sym,
                ctx=old_model.ctx,
                num_epoch=1,
                epoch_size=old_model.epoch_size,
                optimizer='sgd',
                initializer=old_model.initializer,
                numpy_batch_size=old_model.numpy_batch_size,
                arg_params=arg_params,
                aux_params=old_model.aux_params,
                allow_extra_params=True,
                begin_epoch=old_model.begin_epoch)
  return new_model
