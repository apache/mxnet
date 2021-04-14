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
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace MxNet.Gluon.Metrics
{
    public abstract class EvalMetric : Base
    {
        internal long global_num_inst;
        internal float global_sum_metric;

        internal bool hasGlobalStats;

        internal long num_inst;
        internal float sum_metric;

        public EvalMetric(string name, string[] output_names = null, string[] label_names = null,
            bool has_global_stats = false)
        {
            Name = name;
            OutputNames = output_names;
            LabelNames = label_names;
            hasGlobalStats = has_global_stats;
        }

        public EvalMetric(string name, string output_name = null, string label_name = null,
           bool has_global_stats = false)
        {
            Name = name;
            OutputNames = new string[] { output_name };
            LabelNames = new string[] { label_name };
            hasGlobalStats = has_global_stats;
        }

        public string Name { get; internal set; }

        public string[] OutputNames { get; internal set; }

        public string[] LabelNames { get; internal set; }

        public virtual ConfigData GetConfig()
        {
            var config = new ConfigData();
            config.Add("metric", GetType().Name);
            config.Add("name", Name);
            config.Add("output_names", OutputNames);
            config.Add("label_names", LabelNames);

            return config;
        }

        public abstract void Update(ndarray labels, ndarray preds);

        public virtual void Update(NDArrayList labels, NDArrayList preds)
        {
            if (labels.Length != preds.Length) throw new ArgumentException("Labels and Predictions are unequal length");

            for (var i = 0; i < labels.Length; i++) Update(labels[i], preds[i]);
        }

        public void UpdateDict(NDArrayDict labels, NDArrayDict preds)
        {
            if (labels.Count != preds.Count) throw new ArgumentException("Labels and Predictions are unequal length");
            for (int i = 0; i < labels.Count; i++)
            {
                Update(labels[labels.Keys[i]], preds[preds.Keys[i]]);
            }
        }

        public virtual void Reset()
        {
            num_inst = 0;
            global_num_inst = 0;
            sum_metric = 0;
            global_sum_metric = 0;
        }

        public virtual void ResetLocal()
        {
            num_inst = 0;
            sum_metric = 0;
        }

        public virtual (string, float) Get()
        {
            if (num_inst == 0)
                return (Name, float.NaN);

            return (Name, sum_metric / num_inst);
        }

        public virtual (string, float) GetGlobal()
        {
            if (hasGlobalStats)
            {
                if (global_num_inst == 0)
                    return (Name, float.NaN);

                return (Name, global_sum_metric / global_num_inst);
            }

            return Get();
        }

        public Dictionary<string, float> GetNameValue()
        {
            var nameValue = Get();
            return new Dictionary<string, float> {{nameValue.Item1, nameValue.Item2}};
        }

        public Dictionary<string, float> GetGlobalNameValue()
        {
            var nameValue = GetGlobal();
            return new Dictionary<string, float> {{nameValue.Item1, nameValue.Item2}};
        }

        public static implicit operator EvalMetric(string name)
        {
            var assembly = Assembly.GetAssembly(Type.GetType($"MxNet.Gluon.Metrics.EvalMetric"));
            var types = assembly.GetTypes().Where(t => String.Equals(t.Namespace, "MxNet.Gluon.Metrics", StringComparison.Ordinal)).ToList();

            foreach (var item in types)
            {
                var obj = Activator.CreateInstance(item);
                var evalName = item.GetProperty("Name").GetValue(obj);
                if(evalName != null && evalName.ToString().ToLower() == name.ToLower())
                {
                    return  (EvalMetric)obj;
                }
            }

            throw new Exception($"Metric with name '{name}' not found.");
        }
    }
}