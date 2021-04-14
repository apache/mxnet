using MxNet.Gluon.Probability.Distributions.Constraints;
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Bernoulli : ExponentialFamily
    {
        public NDArrayOrSymbol Prob
        {
            get
            {
                return DistributionsUtils.Logit2Prob(this.logit, true);
            }
        }

        public NDArrayOrSymbol Logit
        {
            get
            {
                return DistributionsUtils.Prob2Logit(this.prob, true);
            }
        }

        public override NDArrayOrSymbol Mean
        {
            get
            {
                return this.prob;
            }
        }

        public override NDArrayOrSymbol Variance
        {
            get
            {
                return this.prob * (1 - this.prob);
            }
        }

        public override NDArrayOrSymbolList NaturalParams
        {
            get
            {
                return new NDArrayOrSymbolList(logit);
            }
        }

        public new Dictionary<string, Constraint> arg_constraints = new Dictionary<string, Constraint> 
        {
            {
                "prob",
                new Interval(0, 1)},
            {
                "logit",
                new Real()
            }
        };

        public Bernoulli(NDArrayOrSymbol prob = null, NDArrayOrSymbol logit = null, bool? validate_args = null)
            : base(0, validate_args)
        {
            if (prob != null)
            {
                this.prob = prob;
            }
            else
            {
                this.logit = logit;
            }
        }

        public override Distribution BroadcastTo(Shape batch_shape)
        {
            var new_instance = new Distribution();
            if (this.prob != null)
            {
                new_instance.prob = this.prob.IsNDArray ? nd.BroadcastTo(this.prob, batch_shape) : sym.BroadcastTo(this.prob, batch_shape);
            }
            else
            {
                new_instance.logit = this.logit.IsNDArray ? nd.BroadcastTo(this.logit, batch_shape) : sym.BroadcastTo(this.logit, batch_shape);
            }

            new_instance.event_dim = event_dim;
            new_instance._validate_args = this._validate_args;
            return new_instance;
        }

        public override NDArrayOrSymbol LogProb(NDArrayOrSymbol value)
        {
            if (this._validate_args)
            {
                this.ValidateSamples(value);
            }
            
            if (this.prob == null)
            {
                var logit = this.logit;
                return logit * (value - 1) - F.log(F.exp(F.negative(logit)) + 1);
            }
            else
            {
                // Parameterized by probability
                var eps = 1E-12f;
                return F.log(this.prob + eps) * value + F.log1p(F.negative(this.prob) + eps) * (1 - value);
            }
        }

        public override NDArrayOrSymbol Sample(Shape size)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol SampleN(Shape size)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol LogNormalizer(NDArrayOrSymbol x)
        {
            if (x.IsNDArray)
                return nd.Log(1 + nd.Exp(x));

            return sym.Log(1 + sym.Exp(x));
        }

        public override NDArrayOrSymbol Entropy()
        {
            return F.negative(logit * (prob - 1) - F.log(F.exp(F.negative(logit)) + 1));
        }
    }
}
