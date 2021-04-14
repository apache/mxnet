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
using MxNet.Interop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace MxNet
{
    public struct LibFeature
    {
        public string name;
        public bool enabled;
    }

    public class Runtime
    {
        public static Features FeatureList()
        {
            List<Feature> featuresList = new List<Feature>();
            
            NativeMethods.MXLibInfoFeatures(out var intPtr, out int size);
            int objsize = Marshal.SizeOf(typeof(LibFeature));
            if (size > 0)
            {
                for(int i = 0;i<size;i++)
                {
                    var f = (LibFeature)Marshal.PtrToStructure(intPtr, typeof(LibFeature));
                    intPtr += objsize;
                    featuresList.Add(new Feature() { Enabled = f.enabled, Name = f.name });
                }
            }

            return new Features(featuresList.ToArray());
        }

        public class Feature
        {
            public string Name { get; set; }

            public bool Enabled { get; set; }

            public override string ToString()
            {
                if (Enabled)
                    return string.Format("✔ {0}", Name);
                return string.Format("✖ {0}", Name);
            }
        }

        public class Features
        {
            private readonly List<Feature> _features;

            public Features(params Feature[] features)
            {
                _features = features.ToList();
            }

            public bool IsEnabled(string name)
            {
                var f = _features.FirstOrDefault(x => x.Name == name);
                if (f != null)
                    return f.Enabled;

                return false;
            }
        }
    }
}