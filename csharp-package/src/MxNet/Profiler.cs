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
using MxNet.Interop;
using MxNet.Libs;
using ProfileHandle = System.IntPtr;

namespace MxNet
{
    public class Profiler
    {
        public static ProfileHandle profiler_kvstore_handle;

        public class Task : IDisposable
        {
            internal ProfileHandle handle;

            public Task(Domain domain, string name)
            {
                Name = name;
                Domain = domain;
                NativeMethods.MXProfileCreateTask(domain.handle, name, out var @out);
                handle = @out;
            }

            public string Name { get; set; }

            public Domain Domain { get; set; }

            public void Dispose()
            {
                if (handle != null)
                    NativeMethods.MXProfileDestroyHandle(handle);
            }

            public void Start()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStart(handle);
            }

            public void Stop()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStop(handle);
            }

            public override string ToString()
            {
                return Name;
            }
        }

        public class Frame : IDisposable
        {
            internal ProfileHandle handle;

            public Frame(Domain domain, string name)
            {
                Domain = domain;
                Name = name;
                NativeMethods.MXProfileCreateFrame(domain.handle, name, out var @out);
                handle = @out;
            }

            public string Name { get; set; }

            public Domain Domain { get; set; }

            public void Dispose()
            {
                if (handle != null)
                    NativeMethods.MXProfileDestroyHandle(handle);
            }

            public void Start()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStart(handle);
            }

            public void Stop()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStop(handle);
            }

            public override string ToString()
            {
                return Name;
            }
        }

        public class Event : IDisposable
        {
            internal ProfileHandle handle;

            public Event(string name)
            {
                Name = name;
                NativeMethods.MXProfileCreateEvent(name, out var @out);
                handle = @out;
            }

            public string Name { get; set; }

            public void Dispose()
            {
                if (handle != null)
                    NativeMethods.MXProfileDestroyHandle(handle);
            }

            public void Start()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStart(handle);
            }

            public void Stop()
            {
                if (handle != null)
                    NativeMethods.MXProfileDurationStop(handle);
            }

            public override string ToString()
            {
                return Name;
            }
        }

        public class Counter : IDisposable
        {
            internal ProfileHandle handle;

            public Counter(Domain domain, string name, int? value = null)
            {
                Domain = domain;
                Name = name;
                NativeMethods.MXProfileCreateCounter(domain.handle, name, out var @out);
                handle = @out;

                if (value.HasValue)
                    SetValue(value.Value);
            }

            public string Name { get; set; }

            public Domain Domain { get; set; }

            public void Dispose()
            {
                if (handle != null)
                    NativeMethods.MXProfileDestroyHandle(handle);
            }

            public void SetValue(int value)
            {
                NativeMethods.MXProfileSetCounter(handle, value);
            }

            public void Increment(int delta = 1)
            {
                NativeMethods.MXProfileAdjustCounter(handle, delta);
            }

            public void Decrement(int delta = 1)
            {
                NativeMethods.MXProfileAdjustCounter(handle, -delta);
            }

            public override string ToString()
            {
                return Name;
            }

            public static Counter operator +(Counter c, int delta)
            {
                c.Increment(delta);
                return c;
            }

            public static Counter operator -(Counter c, int delta)
            {
                c.Decrement(delta);
                return c;
            }
        }

        public class Marker
        {
            public Marker(Domain domain, string name)
            {
                Domain = domain;
                Name = name;
            }

            public string Name { get; set; }

            public Domain Domain { get; set; }

            public void Mark(string scope = "process")
            {
                NativeMethods.MXProfileSetMarker(Domain.handle, Name, scope);
            }

            public override string ToString()
            {
                return Name;
            }
        }

        public class Domain
        {
            internal ProfileHandle handle;

            public Domain(string name)
            {
                Name = name;
                NativeMethods.MXProfileCreateDomain(name, out var @out);
                handle = @out;
            }

            public string Name { get; set; }

            public override string ToString()
            {
                return Name;
            }

            public Task NewTask(string name)
            {
                return new Task(this, name);
            }

            public Frame NewFrame(string name)
            {
                return new Frame(this, name);
            }

            public Counter NewCounter(string name)
            {
                return new Counter(this, name);
            }

            public Marker NewMarker(string name)
            {
                return new Marker(this, name);
            }
        }

        public static IEnumerable<string> Scope(string name = "<unk>:", bool append_mode = true)
        {
            name = name.EndsWith(":") ? name : name + ":";
            
            if (append_mode && _current_scope.Get() != "<unk>:")
                name = _current_scope.Get() + name;

            var token = _current_scope.Set(name);
            // Invoke the C API to propagate the profiler scope information to the
            // C++ backend.
            NativeMethods.MXSetProfilerScope(name);
            yield return name;
            _current_scope.Reset(token);
            // Invoke the C API once again to recover the previous scope information.
            NativeMethods.MXSetProfilerScope(_current_scope.Get());
        }

        public static ContextVar<string> _current_scope = new ContextVar<string>("profilerscope", @default: "<unk>:");
    }
}