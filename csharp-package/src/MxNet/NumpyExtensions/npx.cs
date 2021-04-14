using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public partial class npx : ND.Numpy.npx
    {
        public static void set_np(bool shape = true, bool array = true, bool dtype = false)
        {
            MxUtil.SetNp(shape, array, dtype);
        }

        public static void reset_np()
        {
            MxUtil.ResetNp();
        }

        public static Context cpu(int device_id)
        {
            return Context.Cpu(device_id);
        }

        public static Context cpu_pinned(int device_id)
        {
            return Context.CpuPinned(device_id);
        }

        public static Context gpu(int device_id)
        {
            return Context.Gpu(device_id);
        }

        public static (long, long) gpu_memory_info(int device_id)
        {
            return MxUtil.GetGPUMemory(device_id);
        }

        public static Context current_context()
        {
            return Context.CurrentContext;
        }

        public static int num_gpus()
        {
            return MxUtil.GetGPUCount();
        }
    }
}
