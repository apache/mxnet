using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MxNet
{
    public abstract class CustomOpProp
    {
        public bool need_top_grad_;

        public CustomOpProp(bool need_top_grad = true)
        {
            this.need_top_grad_ = need_top_grad;
        }

        public virtual (Shape[], Shape[], Shape[]) InferShape(Shape[] in_shape)
        {
            List<Shape> item2 = new List<Shape>();
            for(int i = 0; i< this.ListOutputs().Length; i++)
            {
                item2.Add(in_shape[0]);
            }

            return (in_shape, item2.ToArray(), new Shape[0]);
        }

        public virtual (DType[], DType[], DType[]) InferType(DType[] in_type)
        {
            List<DType> item2 = new List<DType>();
            for (int i = 0; i < this.ListOutputs().Length; i++)
            {
                item2.Add(in_type[0]);
            }

            List<DType> item3 = new List<DType>();
            for (int i = 0; i < this.ListAuxilaryStates().Length; i++)
            {
                item3.Add(in_type[0]);
            }

            return (in_type, item2.ToArray(), item3.ToArray());
        }

        public virtual (StorageStype[], StorageStype[], StorageStype[]) InferStorageType(StorageStype[] in_stype)
        {
            List<StorageStype> item2 = new List<StorageStype>();
            for (int i = 0; i < this.ListOutputs().Length; i++)
            {
                item2.Add(StorageStype.Default);
            }

            List<StorageStype> item3 = new List<StorageStype>();
            for (int i = 0; i < this.ListAuxilaryStates().Length; i++)
            {
                item3.Add(StorageStype.Default);
            }

            return (in_stype, item2.ToArray(), item3.ToArray());
        }

        public virtual (StorageStype[], StorageStype[], StorageStype[], StorageStype[], StorageStype[]) InferStorageType(StorageStype[] ograd_stype, StorageStype[] in_stype, StorageStype[] out_stype, StorageStype[] igrad_stype, StorageStype[] aux_stype)
        {
            foreach (var _tup_1 in ograd_stype.Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1)))
            {
                var i = _tup_1.Item1;
                var stype = _tup_1.Item2;
                Debug.Assert(stype == StorageStype.Default, $"Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '{stype}' for ograd_stype[{i}]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default output gradient stypes");
            }

            foreach (var _tup_2 in igrad_stype.Select((_p_3, _p_4) => Tuple.Create(_p_4, _p_3)))
            {
                var i = _tup_2.Item1;
                var stype = _tup_2.Item2;
                if (stype == StorageStype.Undefined)
                {
                    stype = StorageStype.Default;
                }

                Debug.Assert(stype == StorageStype.Default, $"Default infer_storage_type_backward implementation doesnt allow non default stypes: found non default stype '{stype}' for igrad_stype[{i}]. Please implement infer_storage_type and infer_storage_type_backward interface in your custom operator if you have non-default input gradient stypes");
            }

            for (int i = 0; i < ograd_stype.Length; i++)
                ograd_stype[i] = StorageStype.Default;

            for (int i = 0; i < in_stype.Length; i++)
                in_stype[i] = StorageStype.Default;

            for (int i = 0; i < out_stype.Length; i++)
                out_stype[i] = StorageStype.Default;

            for (int i = 0; i < igrad_stype.Length; i++)
                igrad_stype[i] = StorageStype.Default;

            for (int i = 0; i < aux_stype.Length; i++)
                aux_stype[i] = StorageStype.Default;

            return (ograd_stype, in_stype, out_stype, igrad_stype, aux_stype);
        }

        public virtual string[] ListOutputs()
        {
            return new string[] { "output" };
        }

        public virtual string[] ListArguments()
        {
            return new string[] { "data" };
        }

        public virtual string[] ListAuxilaryStates()
        {
            return new string[0];
        }

        public virtual int[] DeclareBackwardDependency(int[] out_grad, int[] in_data, int[] out_data)
        {
            var deps = new List<int>();
            if (this.need_top_grad_)
            {
                deps.AddRange(out_grad);
            }

            deps.AddRange(in_data);
            deps.AddRange(out_data);
            return deps.ToArray();
        }
    }
}
