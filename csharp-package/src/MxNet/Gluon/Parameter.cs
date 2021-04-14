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
using System;
using System.Collections.Generic;
using System.Linq;
using MxNet.Initializers;
using MxNet.Numpy;
using MxNet.Sym.Numpy;

namespace MxNet.Gluon
{
    public class DeferredInit
    {
        public Initializer Initializer;

        public Context[] Contexts;

        public Initializer DefaultInit;

        public ndarray Data;

        public DeferredInit(Initializer initializer, Context[] contexts, Initializer default_init, ndarray data)
        {
            Initializer = initializer;
            Contexts = contexts;
            DefaultInit = default_init;
            Data = data;
        }

        public void Deconstruct(out Initializer initializer, out Context[] contexts, out Initializer default_init, out ndarray data)
        {
            initializer = Initializer;
            contexts = Contexts;
            default_init = DefaultInit;
            data = Data;
        }
    }

    public class Parameter
    {
        internal List<Context> _ctx_list;
        internal NDArrayList _data;
        internal NDArrayList _grad;
        internal Shape _shape;
        internal _Symbol _var;
        internal bool allow_deferred_init;
        internal Dictionary<int, List<int>> _ctx_map;
        internal DeferredInit deferred_init;
        internal bool differentiable;
        private OpGradReq grad_req;
        internal Trainer trainer;
        internal string _uuid;
        internal string _var_name;

        public Parameter(string name = "weight", OpGradReq grad_req = OpGradReq.Write, Shape shape = null, DType dtype = null,
            float lr_mult = 1.0f, float wd_mult = 1.0f, Initializer init = null, bool allow_deferred_init = false,
            bool differentiable = true, StorageStype stype = StorageStype.Default,
            StorageStype grad_stype = StorageStype.Default)
        {
            Name = name;
            Lr_Mult = lr_mult;
            Wd_Mult = wd_mult;
            Init = init;
            GradReg = grad_req;
            _shape = shape;
            DataType = dtype ?? DType.Float32;
            this.differentiable = differentiable;
            Stype = stype;
            Grad_Stype = grad_stype;
            this.allow_deferred_init = allow_deferred_init;
            grad_req = OpGradReq.Null;
            this._uuid = Guid.NewGuid().ToString();
            this._var_name = null;
            _ctx_map = new Dictionary<int, List<int>>();
        }

        public virtual OpGradReq GradReg
        {
            get => grad_req;
            set
            {
                if (!differentiable)
                    grad_req = OpGradReq.Null;
                if (grad_req == value)
                    return;

                grad_req = value;
                if (value == OpGradReq.Null && _grad != null)
                {
                    _grad = null;
                    if (_data != null)
                        for (var i = 0; i < _data.Length; i++)
                            _data[i] = _data[i].Detach();
                }
                else if (_data != null)
                {
                    InitGrad();
                }
            }
        }

        public DType DataType { get; internal set; }

        //var data = new List<int>();
        //for (int i = 0; i < data.Length; i++)
        //    data[i] = data[i] != 0 ? data[i] : -1;
        //bool checkZero = true;
        //foreach (var item in _shape.Data.Reverse())
        //{
        //    if (item == 0 && checkZero)
        //        continue;
        //    checkZero = false;
        //    data.Add(item);
        //}
        //data.Reverse();
        //return new Shape(data);
        public Shape Shape => _shape;

        public string Name { get; }

        public float Lr_Mult { get; set; }

        public float Wd_Mult { get; set; }

        public Initializer Init { get; set; }
        public StorageStype Stype { get; }
        public StorageStype Grad_Stype { get; }

        internal void SetTrainer(Trainer trainer)
        {
            if (Stype != StorageStype.Default && this.trainer != null && trainer != null)
                throw new Exception(string.Format(
                    "Failed to set the trainer for Parameter '{0}' because it was already set. " +
                    "More than one trainers for a {1} Parameter is not supported.", Name, Stype));

            this.trainer = trainer;
        }

        internal NDArrayList CheckAndGet(NDArrayList arr_list, Context ctx)
        {
            if (arr_list != null)
            {
                if (ctx == null)
                {
                    if (arr_list.Length == 1)
                        return arr_list.ToArray();
                    else
                        ctx = Context.CurrentContext;
                }

                var ctx_list = _ctx_map[ctx.GetDeviceId()];

                if (ctx.GetDeviceId() < ctx_list.Count)
                {
                    var idx = ctx_list[ctx.GetDeviceId()];
                    if (idx != -1)
                        return arr_list.ToArray();
                }

                throw new Exception($"Parameter '{Name}' was not initialized on context {ctx}. " +
                                        $"It was only initialized on {ctx_list[0]}.");
            }

            if (deferred_init != null)
                throw new DeferredInitializationException(
                    $"Parameter '{Name}' has not been initialized yet because initialization was " +
                    "deferred. Actual initialization happens during the first forward pass. " +
                    "Please pass one batch of data through the network before accessing Parameters. " +
                    "You can also avoid deferred initialization by specifying in_units, " +
                    "num_features, etc., for network layers.");

            throw new ArgumentException($"Parameter '{Name}' has not been initialized. Note that " +
                                        "you should initialize parameters and create Trainer " +
                                        "with Block.collect_params() instead of Block.params " +
                                        "because the later does not include Parameters of " +
                                        "nested child Blocks");
        }

        internal NDArrayList GetRowSparse(NDArrayList arr_list, Context ctx, ndarray row_id)
        {
            if (trainer == null)
                throw new Exception($"Cannot get row_sparse data for Parameter '{Name}' when no " +
                                    "Trainer is created with it.");

            var results = CheckAndGet(arr_list, ctx);
            trainer.RowSparsePull(this, arr_list, row_id);
            return results;
        }

        internal void LoadInit(ndarray data, Context[] ctx, bool cast_dtype = false, string dtype_source = "current")
        {
            if (cast_dtype) Assert.InList("dtype_source", dtype_source, new[] {"current", "saved"});

            if (_shape != null)
            {
                var newshape = new List<int>();
                newshape = Shape.Data.Zip(data.shape.Data, (self_dim, data_dim) =>
                {
                    //ToDo: Review Assert
                    //Assert.InList("self_dim", (int)self_dim, Enumerable.Range(0, (int)data_dim).ToArray(),
                    //            $"Failed loading Parameter '{Name}' from saved params: shape incompatible expected %{Shape} vs saved %{data.Shape}");

                    if (self_dim != 0)
                        return self_dim;
                    return data_dim;
                }).ToList();

                _shape = new Shape(newshape);
            }

            if (DataType != null)
            {
                if (cast_dtype && DataType.Name != data.dtype.Name)
                {
                    if (dtype_source == "current")
                        data = data.AsType(DataType);
                    else if (dtype_source == "saved")
                        DataType = data.dtype;
                }
                else if (DataType.Name != data.dtype.Name)
                {
                    throw new Exception($"Failed loading Parameter '{Name}' from saved params: " +
                                        $"dtype incompatible expected {DataType} vs saved {data.dtype}. " +
                                        "Set cast_dtype=True to cast the dtype of saved params.");
                }

                //ToDo: support for float 16
            }

            if (Stype != data.stype)
                data = data.ToSType(Stype);

            if (_data == null)
            {
                if (deferred_init != null)
                {
                    if (ctx == null || MxUtil.Set(ctx.ToList()) != MxUtil.Set(deferred_init.Contexts.ToList()))
                        throw new Exception($"Failed to load Parameter '{Name}' on {ctx[0]} because it was " +
                                            $"previous initialized on {ListCtx()[0]}.");

                    ctx = deferred_init.Contexts;
                }
                else if (ctx == null)
                {
                    ctx = new[] {Context.Cpu()};
                }

                InitImpl(data, ctx);
            }
            else
            {
                if (ctx == null || MxUtil.Set(ctx.ToList()) != MxUtil.Set(deferred_init.Contexts.ToList()))
                    throw new Exception($"Failed to load Parameter '{Name}' on {ctx[0]} because it was " +
                                        $"previous initialized on {ListCtx()[0]}.");
                SetData(data);
            }

            deferred_init = default;
        }

        internal void FinishDeferredInit()
        {
            if (deferred_init == null)
                return;

            var (init, ctx, default_init, data) = deferred_init;
            deferred_init = null;
            if (!Utils.ShapeIsKnown(Shape))
                throw new Exception($"Cannot initialize Parameter '{Name}' because it has " +
                                    $"invalid shape: {Shape}. Please specify in_units, " +
                                    "in_channels, etc for `Block`s.");

            using (var p = Autograd.Pause())
            {
                if (data == null)
                {
                    data = nd.Zeros(Shape, dtype: DataType, ctx: ctx[0]).ToSType(Stype);

                    if (init == null)
                    {
                        if (default_init != null)
                            default_init.InitWeight(Name, ref data);
                    }
                    else
                    {
                        init.InitWeight(Name, ref data);
                    }

                    InitImpl(data, ctx);
                }
            }
        }

        internal void InitImpl(ndarray data, Context[] ctx_list)
        {
            _ctx_list = ctx_list.ToList();
            _ctx_map = new Dictionary<int, List<int>>();

            for (var i = 0; i < ctx_list.Length; i++)
            {
                var ctx = ctx_list[i];
                var dev_list = _ctx_map.ContainsKey((int)ctx.GetDeviceId() & 1) ? _ctx_map[(int)ctx.GetDeviceId() & 1] : new List<int>();
                
                var key = (int) ctx_list[i].GetDeviceType() & 1;
                while (dev_list.Count <= ctx.GetDeviceId())
                    dev_list.Add(-1);

                dev_list[ctx.GetDeviceId()] = i;
                _ctx_map[i] = dev_list;
                //_ctx_map[key][ctx_list[i].GetDeviceId()] = ctx_list[i];
            }

            _data = new NDArrayList();
            foreach (var ctx in ctx_list) _data.Add(data.AsInContext(ctx));

            InitGrad();
        }

        internal void InitGrad()
        {
            if (GradReg == OpGradReq.Null)
            {
                _grad = null;
                return;
            }

            _grad = new NDArrayList();
            for (var i = 0; i < _data.Length; i++)
                _grad.Add(nd.Zeros(_data[i].shape, _data[i].ctx, _data[i].dtype).ToSType(Stype));

            Autograd.MarkVariables(CheckAndGet(_data, null), _grad.ToArray(), GradReg);
        }

        internal ndarray Reduce()
        {
            var ctx = Context.Cpu();
            ndarray data = null;
            if (Stype == StorageStype.Default)
            {
                var block = ListData();
                if (block.Length > 1)
                    data = nd.AddN(block.Select(x => x.AsInContext(ctx)).ToArray()) / block.Length;
                else
                    data = this.Data().ChangeContext(ctx);
            }
            else
            {
                var all_row_ids = nd.Arange(0, Shape[0], dtype: DType.Int64, ctx: ctx);
                data = nd.Zeros(Shape, ctx).ToSType(StorageStype.RowSparse);
                trainer.RowSparsePull(this, new NDArrayList {data}, all_row_ids, true);
            }

            return data;
        }

        public void Initialize(Initializer init = null, Context[] ctx = null, Initializer default_init = null,
            bool force_reinit = false)
        {
            if (_data != null && !force_reinit)
            {
                Logger.Warning(
                    $"Parameter '{Name}' is already initialized, ignoring. Set force_reinit=True to re-initialize.");
                return;
            }

            _data = null;
            _grad = null;


            if (ctx == null)
                ctx = new[] {Context.CurrentContext};

            if (init == null)
                init = Init != null ? Init : default_init;

            if (!Utils.ShapeIsKnown(Shape))
            {
                if (allow_deferred_init)
                {
                    deferred_init = new DeferredInit(init, ctx, default_init, null);
                    return;
                }

                throw new Exception($"Cannot initialize Parameter '{Name}' because it has invalid shape: {Shape}.");
            }

            deferred_init = new DeferredInit(init, ctx, default_init, null);
            FinishDeferredInit();
        }

        public void ResetCtx(Context ctx)
        {
            if (_data != null)
            {
                var data = Reduce();
                using (var p = Autograd.Pause())
                {
                    InitImpl(data, new Context[] { ctx });
                }
            }
            else if (deferred_init != null)
            {
                var (init, _, default_init, data) = deferred_init;
                deferred_init = new DeferredInit(init, new Context[] { ctx }, default_init, data);
            }
            else
            {
                throw new Exception(
                    $"Cannot reset context for Parameter '{Name}' because it has not been initialized.");
            }
        }

        public void SetData(ndarray data)
        {
            _shape = data.shape;
            if (_data == null)
            {
                if (deferred_init == null)
                    throw new Exception($"Parameter '{Name}' has not been initialized");

                deferred_init = new DeferredInit(deferred_init.Initializer, deferred_init.Contexts, deferred_init.DefaultInit, data);
                return;
            }

            if (trainer != null && trainer._kv_initialized && trainer._update_on_kvstore.Value)
                if (!trainer._params_to_init.Contains(this))
                    trainer.ResetKVstore();

            var dlist = CheckAndGet(_data, null);
            for (var i = 0; i < dlist.Length; i++) dlist[i] = data;
        }

        public ndarray RowSparseData(ndarray row_id)
        {
            if (Stype != StorageStype.RowSparse)
                throw new Exception($"Cannot return copies of Parameter '{Name}' on all contexts via " +
                                    $"list_row_sparse_data() because its storage type is {Stype}. Please " +
                                    "use data() instead.");

            return GetRowSparse(_data, row_id.ctx, row_id).FirstOrDefault();
        }

        public NDArrayList ListRowSparseData(ndarray row_id)
        {
            if (Stype != StorageStype.RowSparse)
                throw new Exception($"Cannot return a copy of Parameter {Name} via row_sparse_data() " +
                                    $"because its storage type is {Stype}. Please use data() instead.");

            return GetRowSparse(_data, null, row_id);
        }

        public ndarray Data(Context ctx = null)
        {
            if (Stype != StorageStype.Default)
                throw new Exception($"Cannot return copies of Parameter '{Name}' on all contexts via " +
                                    $"list_row_sparse_data() because its storage type is {Stype}. Please " +
                                    "use RowSparseData() instead.");

            ndarray data = CheckAndGet(_data, ctx).FirstOrDefault();
            DeferredCompute.SetVariable(data, this.Var());
            return data;
        }

        public NDArrayList ListData()
        {
            if (Stype != StorageStype.Default)
                throw new Exception($"Cannot return a copy of Parameter {Name} via row_sparse_data() " +
                                    $"because its storage type is {Stype}. Please use ListRowSparseData() instead.");

            return CheckAndGet(_data, null);
        }

        public ndarray Grad(Context ctx = null)
        {
            if (_data != null && _grad == null)
                throw new Exception($"Cannot get gradient array for Parameter '{Name}' " +
                                    "because grad_req='null'");

            return CheckAndGet(_grad, ctx).FirstOrDefault();
        }

        public NDArrayList ListGrad()
        {
            if (_data != null && _grad == null)
                throw new Exception($"Cannot get gradient array for Parameter '{Name}' " +
                                    "because grad_req='null'");

            return CheckAndGet(_grad, null);
        }

        public Context[] ListCtx()
        {
            if (_data == null)
            {
                if (deferred_init != null) return deferred_init.Contexts;

                throw new Exception($"Parameter '{Name}' has not been initialized");
            }

            return _ctx_list != null ? _ctx_list.ToArray() : new Context[0];
        }

        public void ZeroGrad()
        {
            if (_grad == null)
                return;

            for (var i = 0; i < _grad.Length; i++)
                _grad[i] = nd.ZerosLike(_grad[i]);
        }

        public _Symbol Var()
        {
            if (_var == null)
            {
                if (string.IsNullOrWhiteSpace(this._var_name))
                {
                    // _var_name is set manually in SymbolBlock.import
                    // The variable name is required by the storage profiler.
                    this._var_name = this._uuid.Replace("-", "_") + "_" + this.Name;
                }
            }
            _var = _Symbol.Var(_var_name, shape: Shape, dtype: DataType, lr_mult: Lr_Mult, wd_mult: Wd_Mult, init: Init,
                stype: Stype);

            return _var;
        }

        public void Cast(DType dtype)
        {
            this._var = null;
            this.DataType = dtype;
            if (this._data == null)
            {
                return;
            }

            using (var ag = Autograd.Pause())
            {
                if (_data != null)
                    for (var i = 0; i < _data.Length; i++)
                        _data[i] = _data[i].Cast(dtype);

                if (_grad != null)
                    for (var i = 0; i < _grad.Length; i++)
                        _data[i] = _data[i].Cast(dtype);

                Autograd.MarkVariables(this._data, this._grad, this.grad_req);
            }
        }


    }
}