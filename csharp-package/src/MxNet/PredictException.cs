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
    /// <summary>
    ///     The exception that is thrown when occurs error of prediction. This class cannot be inherited.
    /// </summary>
    public sealed class PredictException : Exception
    {
        #region Constructors

        /// <summary>
        ///     Initializes a new instance of the <see cref="PredictException" /> class.
        /// </summary>
        public PredictException()
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="PredictException" /> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public PredictException(string message)
            : base(message)
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="PredictException" /> class with a specified error message and a
        ///     reference to the inner exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">The name of the parameter that caused the current exception.</param>
        public PredictException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        #endregion
    }
}