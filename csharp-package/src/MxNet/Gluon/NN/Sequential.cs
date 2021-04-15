/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using System.Collections.Generic;
using System.Linq;

namespace MxNet.Gluon.NN
{
    public class Sequential : Block
    {
        private List<Block> _layers;

        public Sequential() : base()
        {
            _layers = new List<Block>();
        }

        public List<Block> Blocks => _childrens.Values.ToList();

        public new Sequential this[string key]
        {
            get
            {
                var layer = this._childrens[key];
                var net = new Sequential();
                net.Add(layer);
                return net;
            }
        }

        public int Length => _childrens.Count;

        public void Add(params Block[] blocks)
        {
            foreach (var item in blocks)
            {
                _layers.Add(item);
                RegisterChild(item);
            }
        }

        public override NDArrayOrSymbolList Forward(NDArrayOrSymbolList inputs)
        {
            foreach (var block in this._childrens.Values)
            {
                inputs = block.Call(inputs);
            }

            return inputs;
        }

        public override string ToString()
        {
            var modstr = string.Join("\n", _childrens.Select(c => $"  ({c.Key}): {Utils.Indent(c.Value.ToString(), 2)}"));
            return $"{GetType().Name}(\n{modstr}\n)";
        }

        public override void Hybridize(bool active = true, bool partition_if_dynamic = true, bool static_alloc = false, bool static_shape = false, int inline_limit = 2, int? forward_bulk_size = null, int? backward_bulk_size = null)
        {
            if (_childrens.Values.All(x => x.GetType() == typeof(HybridBlock)))
                Logger.Warning(string.Format("All children of this Sequential layer '{0}' are HybridBlocks. Consider " +
                                             "using HybridSequential for the best performance.", Alias()));

            base.Hybridize(active, partition_if_dynamic, static_alloc, static_shape, inline_limit, forward_bulk_size, backward_bulk_size);
        }
    }
}