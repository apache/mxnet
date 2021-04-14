using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public partial class np
    {
        public static ndarray genfromtxt(string fname, NumpyDotNet.dtype dtype = null, string comments = "#", string delimiter = null, int skip_header = 0, 
                                        int skip_footer = 0, Func<float, float> converters = null, string[] missing_values = null, string[] filling_values = null,
                                        int[] usecols = null, string[] names = null, string[] excludelist = null, string deletechars = " !#$%&'()*+, -./:;<=>?@[\\]^{|}~",
                                        string replace_space = "_", bool autostrip = false, bool case_sensitive = true, string defaultfmt = "f%i",
                                        bool? unpack = null, bool usemask = false, bool loose = true, bool invalid_raise = true, int? max_rows = null,
                                        string encoding = "bytes", ndarray like = null)
        {
            throw new NotImplementedException();
        }
    }
}
