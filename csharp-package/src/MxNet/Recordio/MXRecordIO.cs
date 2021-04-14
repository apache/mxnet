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
using System.Diagnostics;
using System.Runtime.InteropServices;
using RecordIOHandle = System.IntPtr;

namespace MxNet.Recordio
{
    public class MXRecordIO : IDisposable
    {
        internal RecordIOHandle handle;

        public string Uri { get; private set; }
        public string Flag { get; private set; }
        public int PID { get; private set; }
        public bool IsOpen { get; private set; }
        public bool Writable { get; private set; }

        public MXRecordIO(string uri, string flag)
        {
            Uri = uri;
            Flag = flag;
            Open();
        }

        public void Dispose()
        {
            Close();
        }

        public virtual void Open()
        {
            if (Flag == "w")
            {
                NativeMethods.MXRecordIOWriterCreate(Uri, out handle);
                Writable = true;
            }
            else if (Flag == "r")
            {
                NativeMethods.MXRecordIOReaderCreate(Uri, out handle);
                Writable = false;
            }

            PID = Process.GetCurrentProcess().Id;
            IsOpen = true;
        }

        public virtual void Close()
        {
            if (!IsOpen)
                return;

            if (Writable)
                NativeMethods.MXRecordIOWriterFree(handle);
            else
                NativeMethods.MXRecordIOReaderFree(handle);
        }

        public virtual void Reset()
        {
            Close();
            Open();
        }

        public virtual void Write(byte[] buf)
        {
            if (!Writable)
                throw new Exception("Not writable!");

            CheckPID(false);
            NativeMethods.MXRecordIOWriterWriteRecord(handle, buf, buf.Length);
        }

        public virtual byte[] Read()
        {
            if (Writable)
                throw new Exception("Not readable!");

            CheckPID(false);
            
            NativeMethods.MXRecordIOReaderReadRecord(handle, out var buff_ptr, out int size);
            unsafe
            {
                Span<byte> byteArray = new Span<byte>(buff_ptr.ToPointer(), size);
                return byteArray.ToArray();
            }
        }

        public virtual Dictionary<string, object> GetState()
        {
            bool is_open = IsOpen;
            Close();
            Dictionary<string, object> d = new Dictionary<string, object>();
            d["is_open"] = is_open;
            if(d.ContainsKey("handle"))
                d.Remove("handle");

            d["uri"] = Uri;
            return d;
        }

        public virtual void SetState(Dictionary<string, object> d)
        {
            var is_open = (bool)d["is_open"];
            IsOpen = false;
            handle = IntPtr.Zero;
            if (is_open)
                Open();
        }

        internal void CheckPID(bool allow_reset = false)
        {
            if (PID != Process.GetCurrentProcess().Id)
                if (allow_reset)
                    Reset();
                else
                    throw new Exception("Forbidden operation in multiple processes");

        }
    }
}