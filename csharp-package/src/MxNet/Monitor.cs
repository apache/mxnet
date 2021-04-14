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
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using MxNet.Interop;
using NDArrayHandle = System.IntPtr;
using StatFunc = System.Func<MxNet.NDArray, MxNet.NDArray>;
using Stat = System.Tuple<int, string, MxNet.NDArray>;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public class Monitor
    {
        #region Constructors

        public Monitor(int interval)
            : this(interval, ".*")
        {
        }

        public Monitor(int interval, string pattern)
            : this(interval, pattern, DefaultMonitorFunc)
        {
        }

        public Monitor(int interval, string pattern, StatFunc statFunc)
        {
            Interval = interval;
            Pattern = pattern;
            StatFunc = statFunc;

            Exes = new List<Executor>();
            Stats = new List<Stat>();
        }

        #endregion

        #region Properties

        protected bool Activated { get; private set; }

        protected int Interval { get; }

        protected string Pattern { get; }

        protected StatFunc StatFunc { get; }

        protected List<Executor> Exes { get; }

        protected int Step { get; private set; }

        protected List<Stat> Stats { get; }

        #endregion

        #region Methods

        public void Install(Executor exe)
        {
            if (exe == null)
                throw new ArgumentNullException(nameof(exe));

            unsafe
            {
                var functionPointer =
                    Marshal.GetFunctionPointerForDelegate(
                        new NativeMethods.ExecutorMonitorCallbackDelegate(executor_callback));
                var gcHandle = GCHandle.Alloc(functionPointer);
                var callbackHandle = GCHandle.Alloc(this);
                var callback = (IntPtr) functionPointer.ToPointer();
                NativeMethods.MXExecutorSetMonitorCallback(exe.Handle, callback, (IntPtr) callbackHandle);
                callbackHandle.Free();
                gcHandle.Free();
            }

            Exes.Add(exe);
        }

        public void Tic()
        {
            if (Step % Interval == 0)
            {
                Activated = true;
                Stats.Clear();
            }
        }

        public Stat[] Toc()
        {
            var results = new List<Stat>();

            if (Activated)
            {
                Activated = false;

                foreach (var exe in Exes)
                {
                    foreach (var arg in exe.ArgmentArrays)
                        arg.WaitToRead();

                    foreach (var aux in exe.AuxiliaryArrays)
                        aux.WaitToRead();

                    foreach (var pair in exe.ArgmentDictionary())
                        if (Regex.IsMatch(pair.Key, Pattern))
                            Stats.Add(new Stat(Step, pair.Key, StatFunc(pair.Value)));

                    foreach (var pair in exe.AuxiliaryDictionary())
                        if (Regex.IsMatch(pair.Key, Pattern))
                            Stats.Add(new Stat(Step, pair.Key, StatFunc(pair.Value)));
                }

                var tmp = results.ToArray();
                results.Clear();
                results.AddRange(Stats);
                Stats.Clear();
                Stats.AddRange(tmp);
            }

            ++Step;

            return results.ToArray();
        }

        public void TocPrint()
        {
            var results = Toc();
            var data = new float[1];
            foreach (var stat in results)
            {
                var ndarray = stat.Item3;

                string str;
                if (ndarray.Size == 1)
                {
                    if (ndarray.GetContext().GetDeviceType() != DeviceType.GPU)
                        unsafe
                        {
                            var p = (float*) ndarray.GetData();
                            data[0] = p[0];
                        }
                    else
                        ndarray.SyncCopyToCPU(data);

                    str = data[0].ToString(CultureInfo.InvariantCulture);
                }
                else
                {
                    str = ndarray.ToString();
                }

                Logging.LG($"Batch: {stat.Item1} {stat.Item2} {str}");
            }
        }

        #region Helpers

        protected static void executor_callback(string name, NDArrayHandle handle, NDArrayHandle monitorPtr)
        {
            var monitor = GCHandle.FromIntPtr(monitorPtr).Target as Monitor;
            if (monitor != null && monitor.Activated && Regex.IsMatch(name, monitor.Pattern))
                monitor.Stats.Add(new Stat(monitor.Step, name, monitor.StatFunc(new NDArray(handle))));
        }

        private static NDArray DefaultMonitorFunc(NDArray x)
        {
            using (var op = new Operator("norm"))
            {
                return op.PushInput(x).Invoke() / (float) Math.Sqrt(x.Size);
            }
        }

        #endregion

        #endregion
    }
}