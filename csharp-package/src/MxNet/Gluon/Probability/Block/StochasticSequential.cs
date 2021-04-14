using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability
{
    public class StochasticSequential : StochasticBlock
    {
        public List<Block> _layers;

        public int Length => throw new NotImplementedRelease1Exception();

        public new StochasticSequential this[string key]
        {
            get
            {
                var layer = this._childrens[key];
                var net = new StochasticSequential();
                net.Add(layer);
                return net;
            }
        }

        public StochasticSequential(Dictionary<string, Block> blocks = null, bool loadkeys = false) : base(blocks, loadkeys)
        {
            this._layers = new List<Block>();
        }

        public void Add(params Block[] blocks)
        {
            foreach (var block in blocks)
            {
                this._layers.Add(block);
                this.RegisterChild(block);
            }
        }

        public override NDArrayOrSymbol Forward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            foreach (var (k, block) in this._childrens)
            {
                x = block.Call(x, args);
                //var newargs = new NDArrayOrSymbolList();
                //if (x is tuple || x is list)
                //{
                //    args = x[1];
                //    x = x[0];
                //}
            }
            //if (args)
            //{
            //    x = tuple(new List<object> {
            //            x
            //        } + args.ToList());
            //}
            foreach (var block in this._layers)
            {
                this.AddLoss(((StochasticBlock)block)._losses);
            }
            return x;
        }
    }
}
