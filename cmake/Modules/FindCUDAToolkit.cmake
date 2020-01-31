# Copyright 2000-2019 Kitware, Inc. and Contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of Kitware, Inc. nor the names of Contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[=======================================================================[.rst:
FindCUDAToolkit
---------------

This script locates the NVIDIA CUDA toolkit and the associated libraries, but
does not require the ``CUDA`` language be enabled for a given project. This
module does not search for the NVIDIA CUDA Samples.

Search Behavior
^^^^^^^^^^^^^^^

Finding the CUDA Toolkit requires finding the ``nvcc`` executable, which is
searched for in the following order:

1. If the ``CUDA`` language has been enabled we will use the directory
   containing the compiler as the first search location for ``nvcc``.

2. If the ``CUDAToolkit_ROOT`` cmake configuration variable (e.g.,
   ``-DCUDAToolkit_ROOT=/some/path``) *or* environment variable is defined, it
   will be searched.  If both an environment variable **and** a
   configuration variable are specified, the *configuration* variable takes
   precedence.

   The directory specified here must be such that the executable ``nvcc`` can be
   found underneath the directory specified by ``CUDAToolkit_ROOT``.  If
   ``CUDAToolkit_ROOT`` is specified, but no ``nvcc`` is found underneath, this
   package is marked as **not** found.  No subsequent search attempts are
   performed.

3. If the CUDA_PATH environment variable is defined, it will be searched.

4. The user's path is searched for ``nvcc`` using :command:`find_program`.  If
   this is found, no subsequent search attempts are performed.  Users are
   responsible for ensuring that the first ``nvcc`` to show up in the path is
   the desired path in the event that multiple CUDA Toolkits are installed.

5. On Unix systems, if the symbolic link ``/usr/local/cuda`` exists, this is
   used.  No subsequent search attempts are performed.  No default symbolic link
   location exists for the Windows platform.

6. The platform specific default install locations are searched.  If exactly one
   candidate is found, this is used.  The default CUDA Toolkit install locations
   searched are:

   +-------------+-------------------------------------------------------------+
   | Platform    | Search Pattern                                              |
   +=============+=============================================================+
   | macOS       | ``/Developer/NVIDIA/CUDA-X.Y``                              |
   +-------------+-------------------------------------------------------------+
   | Other Unix  | ``/usr/local/cuda-X.Y``                                     |
   +-------------+-------------------------------------------------------------+
   | Windows     | ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`` |
   +-------------+-------------------------------------------------------------+

   Where ``X.Y`` would be a specific version of the CUDA Toolkit, such as
   ``/usr/local/cuda-9.0`` or
   ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0``

   .. note::

       When multiple CUDA Toolkits are installed in the default location of a
       system (e.g., both ``/usr/local/cuda-9.0`` and ``/usr/local/cuda-10.0``
       exist but the ``/usr/local/cuda`` symbolic link does **not** exist), this
       package is marked as **not** found.

       There are too many factors involved in making an automatic decision in
       the presence of multiple CUDA Toolkits being installed.  In this
       situation, users are encouraged to either (1) set ``CUDAToolkit_ROOT`` or
       (2) ensure that the correct ``nvcc`` executable shows up in ``$PATH`` for
       :command:`find_program` to find.

Options
^^^^^^^

``VERSION``
    If specified, describes the version of the CUDA Toolkit to search for.

``REQUIRED``
    If specified, configuration will error if a suitable CUDA Toolkit is not
    found.

``QUIET``
    If specified, the search for a suitable CUDA Toolkit will not produce any
    messages.

``EXACT``
    If specified, the CUDA Toolkit is considered found only if the exact
    ``VERSION`` specified is recovered.

Imported targets
^^^^^^^^^^^^^^^^

An :ref:`imported target <Imported targets>` named ``CUDA::toolkit`` is provided.

This module defines :prop_tgt:`IMPORTED` targets for each
of the following libraries that are part of the CUDAToolkit:

- :ref:`CUDA Runtime Library<cuda_toolkit_rt_lib>`
- :ref:`CUDA Driver Library<cuda_toolkit_driver_lib>`
- :ref:`cuBLAS<cuda_toolkit_cuBLAS>`
- :ref:`cuFFT<cuda_toolkit_cuFFT>`
- :ref:`cuRAND<cuda_toolkit_cuRAND>`
- :ref:`cuSOLVER<cuda_toolkit_cuSOLVER>`
- :ref:`cuSPARSE<cuda_toolkit_cuSPARSE>`
- :ref:`NPP<cuda_toolkit_NPP>`
- :ref:`nvBLAS<cuda_toolkit_nvBLAS>`
- :ref:`nvGRAPH<cuda_toolkit_nvGRAPH>`
- :ref:`nvJPEG<cuda_toolkit_nvJPEG>`
- :ref:`nvidia-ML<cuda_toolkit_nvML>`
- :ref:`nvRTC<cuda_toolkit_nvRTC>`
- :ref:`nvToolsExt<cuda_toolkit_nvToolsExt>`
- :ref:`OpenCL<cuda_toolkit_opencl>`
- :ref:`cuLIBOS<cuda_toolkit_cuLIBOS>`

.. _`cuda_toolkit_rt_lib`:

CUDA Runtime Library
""""""""""""""""""""

The CUDA Runtime library (cudart) are what most applications will typically
need to link against to make any calls such as `cudaMalloc`, and `cudaFree`.
They are an explicit dependency of almost every library.

Targets Created:

- ``CUDA::cudart``
- ``CUDA::cudart_static``

.. _`cuda_toolkit_driver_lib`:

CUDA Driver Library
""""""""""""""""""""

The CUDA Driver library (cuda) are used by applications that use calls
such as `cuMemAlloc`, and `cuMemFree`. This is generally used by advanced


Targets Created:

- ``CUDA::cuda_driver``
- ``CUDA::cuda_driver``

.. _`cuda_toolkit_cuBLAS`:

cuBLAS
""""""

The `cuBLAS <https://docs.nvidia.com/cuda/cublas/index.html>`_ library.

Targets Created:

- ``CUDA::cublas``
- ``CUDA::cublas_static``

.. _`cuda_toolkit_cuFFT`:

cuFFT
"""""

The `cuFFT <https://docs.nvidia.com/cuda/cufft/index.html>`_ library.

Targets Created:

- ``CUDA::cufft``
- ``CUDA::cufftw``
- ``CUDA::cufft_static``
- ``CUDA::cufftw_static``

cuRAND
""""""

The `cuRAND <https://docs.nvidia.com/cuda/curand/index.html>`_ library.

Targets Created:

- ``CUDA::curand``
- ``CUDA::curand_static``

.. _`cuda_toolkit_cuSOLVER`:

cuSOLVER
""""""""

The `cuSOLVER <https://docs.nvidia.com/cuda/cusolver/index.html>`_ library.

Targets Created:

- ``CUDA::cusolver``
- ``CUDA::cusolver_static``

.. _`cuda_toolkit_cuSPARSE`:

cuSPARSE
""""""""

The `cuSPARSE <https://docs.nvidia.com/cuda/cusparse/index.html>`_ library.

Targets Created:

- ``CUDA::cusparse``
- ``CUDA::cusparse_static``

.. _`cuda_toolkit_NPP`:

NPP
"""

The `NPP <https://docs.nvidia.com/cuda/npp/index.html>`_ libraries.

Targets Created:

- `nppc`:

  - ``CUDA::nppc``
  - ``CUDA::nppc_static``

- `nppial`: Arithmetic and logical operation functions in `nppi_arithmetic_and_logical_operations.h`

  - ``CUDA::nppial``
  - ``CUDA::nppial_static``

- `nppicc`: Color conversion and sampling functions in `nppi_color_conversion.h`

  - ``CUDA::nppicc``
  - ``CUDA::nppicc_static``

- `nppicom`: JPEG compression and decompression functions in `nppi_compression_functions.h`

  - ``CUDA::nppicom``
  - ``CUDA::nppicom_static``

- `nppidei`: Data exchange and initialization functions in `nppi_data_exchange_and_initialization.h`

  - ``CUDA::nppidei``
  - ``CUDA::nppidei_static``

- `nppif`: Filtering and computer vision functions in `nppi_filter_functions.h`

  - ``CUDA::nppif``
  - ``CUDA::nppif_static``

- `nppig`: Geometry transformation functions found in `nppi_geometry_transforms.h`

  - ``CUDA::nppig``
  - ``CUDA::nppig_static``

- `nppim`: Morphological operation functions found in `nppi_morphological_operations.h`

  - ``CUDA::nppim``
  - ``CUDA::nppim_static``

- `nppist`: Statistics and linear transform in `nppi_statistics_functions.h` and `nppi_linear_transforms.h`

  - ``CUDA::nppist``
  - ``CUDA::nppist_static``

- `nppisu`: Memory support functions in `nppi_support_functions.h`

  - ``CUDA::nppisu``
  - ``CUDA::nppisu_static``

- `nppitc`: Threshold and compare operation functions in `nppi_threshold_and_compare_operations.h`

  - ``CUDA::nppitc``
  - ``CUDA::nppitc_static``

- `npps`:

  - ``CUDA::npps``
  - ``CUDA::npps_static``

.. _`cuda_toolkit_nvBLAS`:

nvBLAS
""""""

The `nvBLAS <https://docs.nvidia.com/cuda/nvblas/index.html>`_ libraries.
This is a shared library only.

Targets Created:

- ``CUDA::nvblas``

.. _`cuda_toolkit_nvGRAPH`:

nvGRAPH
"""""""

The `nvGRAPH <https://docs.nvidia.com/cuda/nvgraph/index.html>`_ library.

Targets Created:

- ``CUDA::nvgraph``
- ``CUDA::nvgraph_static``


.. _`cuda_toolkit_nvJPEG`:

nvJPEG
""""""

The `nvJPEG <https://docs.nvidia.com/cuda/nvjpeg/index.html>`_ library.
Introduced in CUDA 10.

Targets Created:

- ``CUDA::nvjpeg``
- ``CUDA::nvjpeg_static``

.. _`cuda_toolkit_nvRTC`:

nvRTC
"""""

The `nvRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_ (Runtime Compilation) library.
This is a shared library only.

Targets Created:

- ``CUDA::nvrtc``

.. _`cuda_toolkit_nvml`:

nvidia-ML
"""""""""

The `NVIDIA Management Library <https://developer.nvidia.com/nvidia-management-library-nvml>`_.
This is a shared library only.

Targets Created:

- ``CUDA::nvml``

.. _`cuda_toolkit_opencl`:

.. _`cuda_toolkit_nvToolsExt`:

nvToolsExt
""""""""""

The `NVIDIA Tools Extension <https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm>`_.
This is a shared library only.

Targets Created:

- ``CUDA::nvToolsExt``

OpenCL
""""""

The `NVIDIA OpenCL Library <https://developer.nvidia.com/opencl>`_.
This is a shared library only.

Targets Created:

- ``CUDA::OpenCL``

.. _`cuda_toolkit_cuLIBOS`:

cuLIBOS
"""""""

The cuLIBOS library is a backend thread abstraction layer library which is
static only.  The ``CUDA::cublas_static``, ``CUDA::cusparse_static``,
``CUDA::cufft_static``, ``CUDA::curand_static``, and (when implemented) NPP
libraries all automatically have this dependency linked.

Target Created:

- ``CUDA::culibos``

**Note**: direct usage of this target by consumers should not be necessary.

.. _`cuda_toolkit_cuRAND`:



Result variables
^^^^^^^^^^^^^^^^

``CUDAToolkit_FOUND``
    A boolean specifying whether or not the CUDA Toolkit was found.

``CUDAToolkit_VERSION``
    The exact version of the CUDA Toolkit found (as reported by
    ``nvcc --version``).

``CUDAToolkit_VERSION_MAJOR``
    The major version of the CUDA Toolkit.

``CUDAToolkit_VERSION_MAJOR``
    The minor version of the CUDA Toolkit.

``CUDAToolkit_VERSION_PATCH``
    The patch version of the CUDA Toolkit.

``CUDAToolkit_BIN_DIR``
    The path to the CUDA Toolkit library directory that contains the CUDA
    executable ``nvcc``.

``CUDAToolkit_INCLUDE_DIRS``
    The path to the CUDA Toolkit ``include`` folder containing the header files
    required to compile a project linking against CUDA.

``CUDAToolkit_LIBRARY_DIR``
    The path to the CUDA Toolkit library directory that contains the CUDA
    Runtime library ``cudart``.

``CUDAToolkit_NVCC_EXECUTABLE``
    The path to the NVIDIA CUDA compiler ``nvcc``.  Note that this path may
    **not** be the same as
    :variable:`CMAKE_CUDA_COMPILER <CMAKE_<LANG>_COMPILER>`.  ``nvcc`` must be
    found to determine the CUDA Toolkit version as well as determining other
    features of the Toolkit.  This variable is set for the convenience of
    modules that depend on this one.


#]=======================================================================]

# NOTE: much of this was simply extracted from FindCUDA.cmake.

#   James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#   Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
#
#   Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#   Copyright (c) 2007-2009
#   Scientific Computing and Imaging Institute, University of Utah
#
#   This code is licensed under the MIT License.  See the FindCUDA.cmake script
#   for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

if(CMAKE_CUDA_COMPILER_LOADED AND NOT CUDAToolkit_BIN_DIR)
  get_filename_component(cuda_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  # use the already detected cuda compiler
  set(CUDAToolkit_BIN_DIR "${cuda_dir}" CACHE PATH "")
  unset(cuda_dir)
endif()

# Try language- or user-provided path first.
if(CUDAToolkit_BIN_DIR)
  find_program(CUDAToolkit_NVCC_EXECUTABLE
    NAMES nvcc nvcc.exe
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH
    )
endif()

# Search using CUDAToolkit_ROOT
find_program(CUDAToolkit_NVCC_EXECUTABLE
  NAMES nvcc nvcc.exe
  PATHS ENV CUDA_PATH
  PATH_SUFFIXES bin
)

# If the user specified CUDAToolkit_ROOT but nvcc could not be found, this is an error.
if (NOT CUDAToolkit_NVCC_EXECUTABLE AND (DEFINED CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT}))
  # Declare error messages now, print later depending on find_package args.
  set(fail_base "Could not find nvcc executable in path specified by")
  set(cuda_root_fail "${fail_base} CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  set(env_cuda_root_fail "${fail_base} environment variable CUDAToolkit_ROOT=$ENV{CUDAToolkit_ROOT}")

  if (CUDAToolkit_FIND_REQUIRED)
    if (DEFINED CUDAToolkit_ROOT)
      message(FATAL_ERROR ${cuda_root_fail})
    elseif (DEFINED ENV{CUDAToolkit_ROOT})
      message(FATAL_ERROR ${env_cuda_root_fail})
    endif()
  else()
    if (NOT CUDAToolkit_FIND_QUIETLY)
      if (DEFINED CUDAToolkit_ROOT)
        message(STATUS ${cuda_root_fail})
      elseif (DEFINED ENV{CUDAToolkit_ROOT})
        message(STATUS ${env_cuda_root_fail})
      endif()
    endif()
    set(CUDAToolkit_FOUND FALSE)
    unset(fail_base)
    unset(cuda_root_fail)
    unset(env_cuda_root_fail)
    return()
  endif()
endif()

# CUDAToolkit_ROOT cmake / env variable not specified, try platform defaults.
#
# - Linux: /usr/local/cuda-X.Y
# - macOS: /Developer/NVIDIA/CUDA-X.Y
# - Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y
#
# We will also search the default symlink location /usr/local/cuda first since
# if CUDAToolkit_ROOT is not specified, it is assumed that the symlinked
# directory is the desired location.
if (NOT CUDAToolkit_NVCC_EXECUTABLE)
  if (UNIX)
    if (NOT APPLE)
      set(platform_base "/usr/local/cuda-")
    else()
      set(platform_base "/Developer/NVIDIA/CUDA-")
    endif()
  else()
    set(platform_base "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v")
  endif()

  # Build out a descending list of possible cuda installations, e.g.
  file(GLOB possible_paths "${platform_base}*")
  # Iterate the glob results and create a descending list.
  set(possible_versions)
  foreach (p ${possible_paths})
    # Extract version number from end of string
    string(REGEX MATCH "[0-9][0-9]?\\.[0-9]$" p_version ${p})
    if (IS_DIRECTORY ${p} AND p_version)
      list(APPEND possible_versions ${p_version})
    endif()
  endforeach()

  # Cannot use list(SORT) because that is alphabetical, we need numerical.
  # NOTE: this is not an efficient sorting strategy.  But even if a user had
  # every possible version of CUDA installed, this wouldn't create any
  # significant overhead.
  set(versions)
  foreach (v ${possible_versions})
    list(LENGTH versions num_versions)
    # First version, nothing to compare with so just append.
    if (num_versions EQUAL 0)
      list(APPEND versions ${v})
    else()
      # Loop through list.  Insert at an index when comparison is
      # VERSION_GREATER since we want a descending list.  Duplicates will not
      # happen since this came from a glob list of directories.
      set(i 0)
      set(early_terminate FALSE)
      while (i LESS num_versions)
        list(GET versions ${i} curr)
        if (v VERSION_GREATER curr)
          list(INSERT versions ${i} ${v})
          set(early_terminate TRUE)
          break()
        endif()
        math(EXPR i "${i} + 1")
      endwhile()
      # If it did not get inserted, place it at the end.
      if (NOT early_terminate)
        list(APPEND versions ${v})
      endif()
    endif()
  endforeach()

  # With a descending list of versions, populate possible paths to search.
  set(search_paths)
  foreach (v ${versions})
    list(APPEND search_paths "${platform_base}${v}")
  endforeach()

  # Force the global default /usr/local/cuda to the front on Unix.
  if (UNIX)
    list(INSERT search_paths 0 "/usr/local/cuda")
  endif()

  # Now search for nvcc again using the platform default search paths.
  find_program(CUDAToolkit_NVCC_EXECUTABLE
    NAMES nvcc nvcc.exe
    PATHS ${search_paths}
    PATH_SUFFIXES bin
  )

  # We are done with these variables now, cleanup for caller.
  unset(platform_base)
  unset(possible_paths)
  unset(possible_versions)
  unset(versions)
  unset(i)
  unset(early_terminate)
  unset(search_paths)

  if (NOT CUDAToolkit_NVCC_EXECUTABLE)
    if (CUDAToolkit_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find nvcc, please set CUDAToolkit_ROOT.")
    elseif(NOT CUDAToolkit_FIND_QUIETLY)
      message(STATUS "Could not find nvcc, please set CUDAToolkit_ROOT.")
    endif()

    set(CUDAToolkit_FOUND FALSE)
    return()
  endif()
endif()

if(NOT CUDAToolkit_BIN_DIR AND CUDAToolkit_NVCC_EXECUTABLE)
  get_filename_component(cuda_dir "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
  set(CUDAToolkit_BIN_DIR "${cuda_dir}" CACHE PATH "" FORCE)
  unset(cuda_dir)
endif()

if(CUDAToolkit_NVCC_EXECUTABLE AND
   CUDAToolkit_NVCC_EXECUTABLE STREQUAL CMAKE_CUDA_COMPILER)
  # Need to set these based off the already computed CMAKE_CUDA_COMPILER_VERSION value
  # This if statement will always match, but is used to provide variables for MATCH 1,2,3...
  if(CMAKE_CUDA_COMPILER_VERSION MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
    set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(CUDAToolkit_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
else()
  # Compute the version by invoking nvcc
  execute_process (COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
  if(NVCC_OUT MATCHES [=[ V([0-9]+)\.([0-9]+)\.([0-9]+)]=])
    set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(CUDAToolkit_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(CUDAToolkit_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(CUDAToolkit_VERSION  "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  endif()
  unset(NVCC_OUT)
endif()


get_filename_component(CUDAToolkit_ROOT_DIR ${CUDAToolkit_BIN_DIR} DIRECTORY ABSOLUTE)

# Now that we have the real ROOT_DIR, find components inside it.
list(APPEND CMAKE_PREFIX_PATH ${CUDAToolkit_ROOT_DIR})

# Find the include/ directory
find_path(CUDAToolkit_INCLUDE_DIR
  NAMES cuda_runtime.h
)

# And find the CUDA Runtime Library libcudart
find_library(CUDA_CUDART
  NAMES cudart
  PATH_SUFFIXES lib64 lib/x64
)
if (NOT CUDA_CUDART AND NOT CUDAToolkit_FIND_QUIETLY)
  message(STATUS "Unable to find cudart library.")
endif()

unset(CUDAToolkit_ROOT_DIR)
list(REMOVE_AT CMAKE_PREFIX_PATH -1)

#-----------------------------------------------------------------------------
# Perform version comparison and validate all required variables are set.
# MXNET NOTE: This differs from CMake source by ${CMAKE_CURRENT_LIST_DIR}
# replaced with ${CMAKE_ROOT}/Modules
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(CUDAToolkit
  REQUIRED_VARS
    CUDAToolkit_INCLUDE_DIR
    CUDA_CUDART
    CUDAToolkit_NVCC_EXECUTABLE
  VERSION_VAR
    CUDAToolkit_VERSION
)

#-----------------------------------------------------------------------------
# Construct result variables
if(CUDAToolkit_FOUND)
 set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIR})
 get_filename_component(CUDAToolkit_LIBRARY_DIR ${CUDA_CUDART} DIRECTORY ABSOLUTE)
endif()

#-----------------------------------------------------------------------------
# Construct import targets
if(CUDAToolkit_FOUND)

  function(find_and_add_cuda_import_lib lib_name)

    if(ARGC GREATER 1)
      set(search_names ${ARGN})
    else()
      set(search_names ${lib_name})
    endif()

    find_library(CUDA_${lib_name}_LIBRARY
      NAMES ${search_names}
      PATHS ${CUDAToolkit_LIBRARY_DIR}
            ENV CUDA_PATH
      PATH_SUFFIXES nvidia/current lib64 lib/x64 lib
    )

    if (NOT CUDA::${lib_name} AND CUDA_${lib_name}_LIBRARY)
      add_library(CUDA::${lib_name} IMPORTED INTERFACE)
      target_include_directories(CUDA::${lib_name} SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
      target_link_libraries(CUDA::${lib_name} INTERFACE "${CUDA_${lib_name}_LIBRARY}")
    endif()
  endfunction()

  function(add_cuda_link_dependency lib_name)
    foreach(dependency IN LISTS ${ARGN})
      target_link_libraries(CUDA::${lib_name} INTERFACE CUDA::${dependency})
    endforeach()
  endfunction()

  add_library(CUDA::toolkit IMPORTED INTERFACE)
  target_include_directories(CUDA::toolkit SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
  target_link_directories(CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")


  find_and_add_cuda_import_lib(cuda_driver cuda)

  find_and_add_cuda_import_lib(cudart)
  find_and_add_cuda_import_lib(cudart_static)

  foreach (cuda_lib cublas cufft cufftw curand cusolver cusparse nvgraph nvjpeg)
    find_and_add_cuda_import_lib(${cuda_lib})
    add_cuda_link_dependency(${cuda_lib} cudart)

    find_and_add_cuda_import_lib(${cuda_lib}_static)
    add_cuda_link_dependency(${cuda_lib}_static cudart_static)
  endforeach()

  # cuSOLVER depends on cuBLAS, and cuSPARSE
  add_cuda_link_dependency(cusolver cublas cusparse)
  add_cuda_link_dependency(cusolver_static cublas_static cusparse)

  # nvGRAPH depends on cuRAND, and cuSOLVER.
  add_cuda_link_dependency(nvgraph curand cusolver)
  add_cuda_link_dependency(nvgraph_static curand_static cusolver_static)

  find_and_add_cuda_import_lib(nppc)
  find_and_add_cuda_import_lib(nppc_static)

  add_cuda_link_dependency(nppc cudart)
  add_cuda_link_dependency(nppc_static cudart_static culibos)

  # Process the majority of the NPP libraries.
  foreach (cuda_lib nppial nppicc nppidei nppif nppig nppim nppist nppitc npps nppicom nppisu)
    find_and_add_cuda_import_lib(${cuda_lib})
    find_and_add_cuda_import_lib(${cuda_lib}_static)
    add_cuda_link_dependency(${cuda_lib} nppc)
    add_cuda_link_dependency(${cuda_lib}_static nppc_static)
  endforeach()

  find_and_add_cuda_import_lib(nvrtc)
  add_cuda_link_dependency(nvrtc cuda_driver)

  find_and_add_cuda_import_lib(nvml nvidia-ml nvml)

  if(WIN32)
    # nvtools can be installed outside the CUDA toolkit directory
    # so prefer the NVTOOLSEXT_PATH windows only environment variable
    # In addition on windows the most common name is nvToolsExt64_1
    find_library(CUDA_nvToolsExt_LIBRARY
      NAMES nvToolsExt64_1 nvToolsExt64 nvToolsExt
      PATHS ENV NVTOOLSEXT_PATH
            ENV CUDA_PATH
      PATH_SUFFIXES lib/x64 lib
    )
  endif()
  find_and_add_cuda_import_lib(nvToolsExt nvToolsExt nvToolsExt64)

  add_cuda_link_dependency(nvToolsExt cudart)

  find_and_add_cuda_import_lib(OpenCL)

  find_and_add_cuda_import_lib(culibos)
  if(TARGET CUDA::culibos)
    foreach (cuda_lib cublas cufft cusparse curand nvjpeg)
      add_cuda_link_dependency(${cuda_lib}_static culibos)
    endforeach()
  endif()

endif()
