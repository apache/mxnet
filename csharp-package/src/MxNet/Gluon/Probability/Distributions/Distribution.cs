using MxNet.Gluon.Probability.Distributions.Constraints;
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Distribution
    {
        public Dictionary<string, object> dict = new Dictionary<string, object>();
        public bool _validate_args = false;
        public int? event_dim;
        public bool has_grad = false;
        public bool has_enumerate_support = false;
        public NDArrayOrSymbol prob;
        public NDArrayOrSymbol logit;

        public Dictionary<string, Constraint> arg_constraints = new Dictionary<string, Constraint>();

        public object this[string key]
        {
            get
            {
                return dict.ContainsKey(key) ? dict[key] : null;
            }
            set
            {
                if (dict.ContainsKey(key))
                    dict[key] = value;
                else
                    dict.Add(key, value);
            }
        }
        public virtual NDArrayOrSymbolList ArgConstraints
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public virtual NDArrayOrSymbol Mean
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public virtual NDArrayOrSymbol Variance
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public virtual NDArrayOrSymbol StdDev
        {
            get
            {
                return Variance.IsNDArray ? nd.Sqrt(this.Variance) : sym.Sqrt(this.Variance);
            }
        }

        public virtual Constraint Support
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public void SetDefaultValidateArgs(bool value)
        {
            _validate_args = value;
        }

        public Distribution(int? event_dim = null, bool? validate_args = null)
        {
            this.event_dim = event_dim;
            if (validate_args != null)
            {
                this._validate_args = validate_args.Value;
            }

            if (this._validate_args)
            {
                foreach (var (param, constraint) in this.arg_constraints)
                {
                    if (!this.dict.ContainsKey(param) && this[param] is _CachedProperty)
                    {
                        // skip param that is decorated by cached_property
                        continue;
                    }

                    this[param] = constraint.Check((NDArrayOrSymbol)this[param]);
                }
            }
        }

        public virtual NDArrayOrSymbol LogProb(NDArrayOrSymbol value)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol Pdf(NDArrayOrSymbol value)
        {
            if(value.IsNDArray)
                return nd.Exp(this.LogProb(value));

            return sym.Exp(this.LogProb(value));
        }

        public virtual NDArrayOrSymbol Cdf(NDArrayOrSymbol value)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol Icdf(NDArrayOrSymbol value)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol Sample(Shape size)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol SampleN(Shape size)
        {
            throw new NotSupportedException();
        }

        public virtual Distribution BroadcastTo(Shape batch_shape)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol EnumerateSupport()
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol Entropy()
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol Perplexity()
        {
            var entropy = this.Entropy();
            if (entropy.IsNDArray)
                return nd.Exp(entropy);

            return sym.Exp(entropy);
        }

        public override string ToString()
        {
            var args_string = "";
            foreach (var (k, _) in this.arg_constraints)
            {
                Shape shape_v = null;
                try
                {
                    var v = this.dict[k];
                }
                catch (Exception)
                {
                    // TODO: Some of the keys in `arg_constraints` are cached_properties, which
                    // are set as instance property only after they are called (hence won't
                    // be in self.__dict__). In case they have not been called yet, we set shape
                    // to `None` - as a quick fix, since it is not known.
                    shape_v = null;
                }

                args_string += $"{k}: size {shape_v}, ";
            }

            args_string += $", Event Dim: {event_dim}";
            return this.GetType().Name + "(" + args_string + ")";
        }

        public ndarray ValidateSamples(ndarray value)
        {
            return this.Support.Check(value);
        }
    }
}
