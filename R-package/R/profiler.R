# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# profiler setting methods
#

#' @export
MX.PROF.STATE <- list(STOP = 0L, RUN = 1L)

#' Set up the configuration of profiler.
#'
#' @param flags list of key/value pair tuples. Indicates configuration parameters
#'              profile_symbolic : boolean, whether to profile symbolic operators
#'              profile_imperative : boolean, whether to profile imperative operators
#'              profile_memory : boolean, whether to profile memory usage
#'              profile_api : boolean, whether to profile the C API
#'              file_name : string, output file for profile data
#'              continuous_dump : boolean, whether to periodically dump profiling data to file
#'              dump_period : float, seconds between profile data dumps
#' @export
mx.profiler.config <- function(params) {
	mx.internal.profiler.config(params)
}

#' Set up the profiler state to record operator.
#'
#' @param state  Indicting whether to run the profiler, can be 'MX.PROF.STATE$RUN' or 'MX.PROF.STATE$STOP'. Default is `MX.PROF.STATE$STOP`.
#' @param filename The name of output trace file. Default is 'profile.json'
#'
#' @export
mx.profiler.state <- function(state = MX.PROF.STATE$STOP) {
	mx.internal.profiler.state(state)
}
