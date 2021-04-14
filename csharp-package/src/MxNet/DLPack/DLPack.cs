using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class DLPack
    {
        public static void DLPackDeleter(IntPtr ptr)
        {
            throw new NotImplementedException();
        }

        public static NDArray NDArrayFromDlpack(DLTensor dlpack)
        {
            throw new NotImplementedException();
        }

        public static DLTensor NDArrayToDlpackRead(NDArray data)
        {
            throw new NotImplementedException();
        }

        public static DLTensor NDArrayToDlpackWrite(NDArray data)
        {
            throw new NotImplementedException();
        }

        public static Func<ndarray, bool, DLTensor> NDArrayFromNumpy = (x, zero_copy) =>
        {
            throw new NotImplementedException();
        };

        public static Dictionary<string, (int, int, int)> TYPE_MAP = new Dictionary<string, (int, int, int)> {
            {
                "int32",
                (0, 32, 1)},
            {
                "int64",
                (0, 64, 1)},
            {
                "bool",
                (1, 1, 1)},
            {
                "uint8",
                (1, 8, 1)},
            {
                "uint32",
                (1, 32, 1)},
            {
                "uint64",
                (1, 64, 1)},
            {
                "float16",
                (2, 16, 1)},
            {
                "float32",
                (2, 32, 1)},
            {
                "float64",
                (2, 64, 1)
            }
        };
    }
}
