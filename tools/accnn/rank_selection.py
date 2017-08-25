import numpy as np
import mxnet as mx
import json
import utils
import math
import sys

def calc_complexity(ishape, node):
  y, x = map(int, eval(node['param']['kernel']))
  N = int(node['param']['num_filter'])
  C, Y, X = ishape  
  return x*(N+C)*X*Y, x*y*N*C*X*Y

def calc_eigenvalue(model, node):
  W = model.arg_params[node['name'] + '_weight'].asnumpy()
  N, C, y, x = W.shape  
  W = W.transpose((1,2,0,3)).reshape((C*y, -1))
  U, D, Q = np.linalg.svd(W, full_matrices=False)
  return D

def get_ranksel(model, ratio):  
  conf = json.loads(model.symbol.tojson())
  _, output_shapes, _ = model.symbol.get_internals().infer_shape(data=(1,3,224,224))
  out_names = model.symbol.get_internals().list_outputs()    
  out_shape_dic = dict(zip(out_names, output_shapes)) 
  nodes = conf['nodes']
  nodes = utils.topsort(nodes)
  C = []
  D = []
  S = []
  conv_names = []
  EC = 0
  for node in nodes:
    if node['op'] == 'Convolution':        
      input_nodes = [nodes[int(j[0])] for j in node['inputs']]
      data = [input_node for input_node in input_nodes\
                                  if not input_node['name'].startswith(node['name'])][0]      

      if utils.is_input(data):
        ishape = (3, 224, 224)
      else:
        ishape = out_shape_dic[data['name'] + '_output'][1:]
      C.append(calc_complexity(ishape, node))
      D.append(int(node['param']['num_filter']))
      S.append(calc_eigenvalue(model, node))
      conv_names.append(node['name'])
      EC += C[-1][1]  
  for s in S:
    ss = sum(s)
    for i in xrange(1, len(s)):
      s[i] += s[i-1]      
  n = len(C)
  EC /= ratio
  dp = [{}, {}]
  dpc = [{} for _ in xrange(n)]
  now, nxt = 0, 1
  dp[now][0] = 0
  for i in xrange(n):
    dp[nxt] = {}    
    sys.stdout.flush()
    for now_c, now_v in dp[now].items():
      for d in xrange(min(len(S[i]), D[i])):
        nxt_c = now_c + (d+1)*C[i][0]
        if nxt_c > EC:
          continue
        nxt_v = dp[now][now_c] + math.log(S[i][d])                
        if dp[nxt].has_key(nxt_c):
          if nxt_v > dp[nxt][nxt_c]:
            dp[nxt][nxt_c] = nxt_v
            dpc[i][nxt_c] = (d,now_c)
        else:
          dp[nxt][nxt_c] = nxt_v
          dpc[i][nxt_c] = (d,now_c)
    now, nxt = nxt, now    
  maxv = -1e9
  target_c = 0
  for c,v in dp[now].items():
    assert c <= EC, 'False'    
    if v > maxv:
      maxv = v
      target_c = c  
  res = [0]*n
  nowc = target_c
  for i in xrange(n-1,-1,-1):    
    res[i] = dpc[i][nowc][0] + 1
    nowc = dpc[i][nowc][1]
  return dict(zip(conv_names, res))
