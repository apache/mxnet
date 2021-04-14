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
using System.Runtime.InteropServices;
using MxNet.Interop;
using SymbolHandle = System.IntPtr;
using NDArrayHandle = System.IntPtr;
using mx_uint = System.UInt32;
using MxNet.Numpy;
using MxNet.Sym.Numpy;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public sealed class Operator : DisposableMXNetObject
    {
        #region Fields

        private static readonly OpMap OpMap;

        private readonly Dictionary<string, string> _KwParams = new Dictionary<string, string>();
        private readonly List<string> _ArgParams = new List<string>();

        private readonly List<SymbolHandle> _InputSymbols = new List<SymbolHandle>();

        private readonly List<NDArrayHandle> _InputNdarrays = new List<NDArrayHandle>();

        private readonly List<string> _InputKeys = new List<string>();

        private readonly List<string> _ArgNames = new List<string>();

        private readonly SymbolHandle _Handle;

        #endregion

        #region Constructors

        static Operator()
        {
            OpMap = new OpMap();
        }

        public Operator(SymbolHandle handle)
        {
            NativePtr = handle;
        }

        public Operator(string operatorName)
        {
            _OpName = operatorName;
            _Handle = OpMap.GetSymbolCreator(operatorName);

            IntPtr return_type = new SymbolHandle();
            Logging.CHECK_EQ(NativeMethods.MXSymbolGetAtomicSymbolInfo(_Handle,
                out var name,
                out var description,
                out var numArgs,
                out var argNames,
                out var argTypeInfos,
                out var argDescriptions,
                out var keyVarNumArgs,
                ref return_type), NativeMethods.OK);

            var argNamesArray = InteropHelper.ToPointerArray(argNames, numArgs);
            for (var i = 0; i < numArgs; ++i)
            {
                var pArgName = argNamesArray[i];
                _ArgNames.Add(Marshal.PtrToStringAnsi(pArgName));
            }
        }

        #endregion

        #region Properties

        public SymbolHandle Handle => NativePtr;

        private readonly string _OpName;

        public string Name
        {
            get
            {
                ThrowIfDisposed();
                return _OpName;
            }
        }

        #endregion

        #region Methods

        public _Symbol CreateSymbol(string name = "")
        {
            if (_InputKeys.Count > 0)
                Logging.CHECK_EQ(_InputKeys.Count, _InputSymbols.Count);

            var pname = name == "" ? null : name;

            var keys = _KwParams.Keys.ToArray();
            var paramKeys = new string[keys.Length];
            var paramValues = new string[keys.Length];
            for (var index = 0; index < keys.Length; index++)
            {
                var key = keys[index];
                paramKeys[index] = key;
                paramValues[index] = _KwParams[key];
            }

            var inputKeys = _InputKeys.Count != 0 ? _InputKeys.ToArray() : null;

            Logging.CHECK_EQ(NativeMethods.MXSymbolCreateAtomicSymbol(_Handle,
                (uint)paramValues.Length,
                paramKeys,
                paramValues,
                out var symbolHandle), NativeMethods.OK);

            Logging.CHECK_EQ(NativeMethods.MXSymbolCompose(symbolHandle,
                pname,
                (uint) _InputSymbols.Count,
                inputKeys,
                _InputSymbols.ToArray()), NativeMethods.OK);

            return new _Symbol(symbolHandle);
        }

        internal _Symbol CreateNpSymbol(string name = "")
        {
            if (_InputKeys.Count > 0)
                Logging.CHECK_EQ(_InputKeys.Count, _InputSymbols.Count);

            var pname = name == "" ? null : name;

            var keys = _KwParams.Keys.ToArray();
            var paramKeys = new string[keys.Length];
            var paramValues = new string[keys.Length];
            for (var index = 0; index < keys.Length; index++)
            {
                var key = keys[index];
                paramKeys[index] = key;
                paramValues[index] = _KwParams[key];
            }

            var inputKeys = _InputKeys.Count != 0 ? _InputKeys.ToArray() : null;

            Logging.CHECK_EQ(NativeMethods.MXSymbolCreateAtomicSymbol(_Handle,
                (uint)paramValues.Length,
                paramKeys,
                paramValues,
                out var symbolHandle), NativeMethods.OK);

            Logging.CHECK_EQ(NativeMethods.MXSymbolCompose(symbolHandle,
                pname,
                (uint)_InputSymbols.Count,
                inputKeys,
                _InputSymbols.ToArray()), NativeMethods.OK);

            return new _Symbol(symbolHandle);
        }

        public ndarray Invoke()
        {
            var output = new ndarray();
            Invoke(output);
            return output;
        }

        public void Invoke(NDArrayList outputs)
        {
            var paramKeys = new List<string>();
            var paramValues = new List<string>();

            foreach (var data in _KwParams)
            {
                paramKeys.Add(data.Key);
                paramValues.Add(data.Value);
            }

            var numInputs = _InputNdarrays.Count;
            var numOutputs = outputs.Length;

            var outputHandles = outputs.Select(s => s.GetHandle()).ToArray();
            var outputsReceiver = IntPtr.Zero;
            GCHandle? gcHandle = null;
            try
            {
                if (outputs.Length > 0)
                {
                    gcHandle = GCHandle.Alloc(outputHandles, GCHandleType.Pinned);
                    outputsReceiver = gcHandle.Value.AddrOfPinnedObject();
                }

                NDArrayHandle[] outputsReceivers = { outputsReceiver };

                Logging.CHECK_EQ(NativeMethods.MXImperativeInvoke(_Handle, numInputs, _InputNdarrays.ToArray(), ref numOutputs,
                    ref outputsReceiver,
                    paramKeys.Count, paramKeys.ToArray(), paramValues.ToArray()), NativeMethods.OK);

                if (outputs.Length > 0)
                {
                    gcHandle?.Free();
                    return;
                }

                outputHandles = new NDArrayHandle[numOutputs];

                Marshal.Copy(outputsReceiver, outputHandles, 0, numOutputs);

                foreach (var outputHandle in outputHandles) outputs.Add(new ndarray(outputHandle));

                gcHandle?.Free();
            }
            catch(Exception ex)
            {
                throw ex;
            }
            finally
            {
                gcHandle?.Free();
            }
        }

        public void Invoke(List<ndarray> outputs)
        {
            var paramKeys = new List<string>();
            var paramValues = new List<string>();

            foreach (var data in _KwParams)
            {
                paramKeys.Add(data.Key);
                paramValues.Add(data.Value);
            }

            var numInputs = _InputNdarrays.Count;
            var numOutputs = outputs.Count;

            var outputHandles = outputs.Select(s => s.GetHandle()).ToArray();
            var outputsReceiver = IntPtr.Zero;
            GCHandle? gcHandle = null;
            try
            {
                if (outputs.Count > 0)
                {
                    gcHandle = GCHandle.Alloc(outputHandles, GCHandleType.Pinned);
                    outputsReceiver = gcHandle.Value.AddrOfPinnedObject();
                }

                NDArrayHandle[] outputsReceivers = { outputsReceiver };

                Logging.CHECK_EQ(NativeMethods.MXImperativeInvoke(_Handle, numInputs, _InputNdarrays.ToArray(), ref numOutputs,
                    ref outputsReceiver,
                    paramKeys.Count, paramKeys.ToArray(), paramValues.ToArray()), NativeMethods.OK);

                if (outputs.Count > 0)
                {
                    gcHandle?.Free();
                    return;
                }

                outputHandles = new NDArrayHandle[numOutputs];

                Marshal.Copy(outputsReceiver, outputHandles, 0, numOutputs);

                foreach (var outputHandle in outputHandles) outputs.Add(new ndarray(outputHandle));

                gcHandle?.Free();
            }
            catch (Exception ex)
            {
                throw ex;
            }
            finally
            {
                gcHandle?.Free();
            }
        }

        public void PushInput(Symbol symbol)
        {
            if (symbol == null)
                throw new ArgumentNullException(nameof(symbol));

            _InputSymbols.Add(symbol.GetHandle());
        }

        public Operator PushInput(NDArray ndarray)
        {
            if (ndarray == null)
                throw new ArgumentNullException(nameof(ndarray));

            _InputNdarrays.Add(ndarray.GetHandle());
            return this;
        }

        public Operator Set(params object[] args)
        {
            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                if (arg is Symbol)
                    SetParam(i, (Symbol)arg);
                else if (arg is _Symbol)
                    SetParam(i, (_Symbol)arg);
                else if (arg is NDArray)
                    SetParam(i, (NDArray)arg);
                else if (arg is ndarray)
                    SetParam(i, (ndarray)arg);
                else if (arg is IEnumerable<Symbol>)
                    SetParam(i, (IEnumerable<Symbol>)arg);
                else if (arg is NDArrayList)
                    SetParam(i, ((NDArrayList)arg).AsEnumerable());
                else
                    SetParam(i, arg);
            }

            return this;
        }

        public Operator SetInput(string name, Symbol symbol)
        {
            if (symbol == null)
                return this;

            _InputKeys.Add(name);
            _InputSymbols.Add(symbol.GetHandle());
            return this;
        }

        public Operator SetInput(SymbolList symbols)
        {
            foreach (var item in symbols) _InputSymbols.Add(item.GetHandle());

            return this;
        }

        public Operator SetInput(string name, NDArray ndarray)
        {
            if (ndarray == null)
                return this;

            _InputKeys.Add(name);
            _InputNdarrays.Add(ndarray.NativePtr);
            return this;
        }

        public Operator SetInput(NDArrayList ndlist)
        {
            foreach (var item in ndlist.Handles) _InputNdarrays.Add(item);

            return this;
        }

        public Operator SetParam(string key, object value)
        {
            if (value == null) return this;

            _KwParams[key] = value.ToValueString();
            return this;
        }

        public Operator SetParam(string key, float[] value)
        {
            if (value == null) return this;

            _KwParams[key] = value.ToValueString();
            return this;
        }

        public Operator SetParam(int pos, NDArray val)
        {
            _InputNdarrays.Add(val.NativePtr);
            return this;
        }

        public Operator SetParam(int pos, ndarray val)
        {
            _InputNdarrays.Add(val.NativePtr);
            return this;
        }

        public Operator SetParam(int pos, Symbol val)
        {
            _InputSymbols.Add(val.GetHandle());
            return this;
        }

        public Operator SetParam(int pos, _Symbol val)
        {
            _InputNdarrays.Add(val.NativePtr);
            return this;
        }

        public Operator SetParam(int pos, IEnumerable<Symbol> val)
        {
            _InputSymbols.AddRange(val.Select(s => s.GetHandle()));
            return this;
        }

        public Operator SetParam(int pos, IEnumerable<ndarray> val)
        {
            _InputSymbols.AddRange(val.Select(s => s.GetHandle()));
            return this;
        }

        public Operator SetParam(int pos, object val)
        {
            _ArgParams.Add(val != null ? val.ToString() : null);
            //_KwParams[_ArgNames[pos]] = val != null ? val.ToString() : null;
            return this;
        }

        #region Overrides

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            NativeMethods.MXSymbolFree(NativePtr);
        }

        #endregion

        #endregion
    }
}