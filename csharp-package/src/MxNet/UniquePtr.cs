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

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public sealed class UniquePtr<T> : IDisposable
    {
        #region Constructors

        public UniquePtr(T obj)
        {
            Ptr = obj;
        }

        #endregion

        #region Methods

        public static void Move(UniquePtr<T> source, out UniquePtr<T> target)
        {
            target = new UniquePtr<T>(source.Ptr);

            source.IsOwner = false;
            target.IsOwner = true;
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Gets a value indicating whether this object is already disposed.
        /// </summary>
        public bool IsDisposed { get; private set; }

        public bool IsOwner { get; private set; } = true;

        public T Ptr { get; }

        #endregion

        #region IDisposable Members

        /// <summary>
        ///     Releases all resources used by this <see cref="DisposableMXNetObject" />.
        /// </summary>
        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        /// <summary>
        ///     Releases all resources used by this <see cref="DisposableMXNetObject" />.
        /// </summary>
        /// <param name="disposing">Indicate value whether <see cref="IDisposable.Dispose" /> method was called.</param>
        private void Dispose(bool disposing)
        {
            if (IsDisposed) return;

            IsDisposed = true;

            if (disposing)
                if (IsOwner)
                    (Ptr as IDisposable)?.Dispose();
        }

        #endregion
    }
}