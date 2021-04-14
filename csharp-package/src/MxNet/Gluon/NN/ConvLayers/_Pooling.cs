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
namespace MxNet.Gluon.NN
{
    public abstract class _Pooling : HybridBlock
    {
        public _Pooling(int[] pool_size, int[] strides, int[] padding, bool ceil_mode, bool global_pool,
            PoolingType pool_type, string layout, bool? count_include_pad = null) : base()
        {
            Kernel = pool_size;
            Strides = strides ?? pool_size;
            Padding = padding;
            CeilMode = ceil_mode;
            GlobalPool = global_pool;
            PoolType = pool_type;
            Layout = layout;
            CountIncludePad = count_include_pad;
        }

        public int[] Kernel { get; set; }

        public int[] Strides { get; set; }

        public int[] Padding { get; set; }

        public bool CeilMode { get; set; }

        public bool GlobalPool { get; set; }

        public PoolingType PoolType { get; set; }

        public string Layout { get; set; }

        public bool? CountIncludePad { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            if (x.IsNDArray)
                return nd.Pooling(x, new Shape(Kernel), PoolType, GlobalPool, stride: new Shape(Strides),
                    pad: new Shape(Padding),
                    count_include_pad: CountIncludePad, layout: Layout, 
                    pooling_convention: CeilMode ? PoolingConvention.Full : PoolingConvention.Valid);

            return sym.Pooling(x, new Shape(Kernel), PoolType, GlobalPool, stride: new Shape(Strides),
                pad: new Shape(Padding),
                count_include_pad: CountIncludePad, layout: Layout, pooling_convention: CeilMode ? PoolingConvention.Full : PoolingConvention.Valid, symbol_name: "fwd");
        }

        public override string Alias()
        {
            return "pool";
        }

        public override string ToString()
        {
            return $"{GetType().Name}(size=({string.Join(", ", Kernel)}), stride=({string.Join(", ", Strides)})" +
                $", padding=({string.Join(", ", Padding)}), ceil_mode={CeilMode}" +
                $", global_pool={GlobalPool}, pool_type={PoolType}, layout={Layout})";
        }
    }
}