using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability
{
    public class StochasticBlock : HybridBlock
    {
        public bool _flag;

        public NDArrayOrSymbolList _losscache;

        public NDArrayOrSymbolList _losses;

        public NDArrayOrSymbolList Losses
        {
            get
            {
                return this._losses;
            }
        }

        public StochasticBlock(Dictionary<string, Block> blocks, bool loadkeys = false) : base(blocks, loadkeys)
        {
            this._losses = new NDArrayOrSymbolList();
            this._losscache = new NDArrayOrSymbolList();
            // Recording whether collectLoss is invoked.
            this._flag = false;
        }

        public void AddLoss(NDArrayOrSymbolList loss)
        {
            this._losscache.Add(loss);
        }

        public Func<FuncArgs, (NDArrayOrSymbol, NDArrayOrSymbolList)> CollectLoss(Func<FuncArgs, NDArrayOrSymbol> func)
        {
            Func<FuncArgs, (NDArrayOrSymbol, NDArrayOrSymbolList)> inner = (kwargs) => {
                // Loss from hybrid_forward
                var func_out = func(kwargs);
                var collected_loss = this._losscache;
                this._losscache = new NDArrayOrSymbolList();
                this._flag = true;
                return (func_out, collected_loss);
            };

            return inner;
        }

        public override NDArrayOrSymbolList Call(NDArrayOrSymbolList args)
        {
            this._flag = false;
            var @out = base.Call(args);
            if (!this._flag)
            {
                throw new Exception("The forward function should be decorated by " + "StochasticBlock.collectLoss");
            }

            this._losses = @out[1];
            return @out[0];
        }
    }
}
