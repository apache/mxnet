# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindFortran
-----------

Finds a fortran compiler, support libraries and companion C/CXX compilers if any.

The module can be used when configuring a project or when running
in cmake -P script mode.

The module may be used multiple times to find different compilers.

Input Variables
^^^^^^^^^^^^^^^

These variables may be set to choose which compiler executable is looked up.

.. variable:: Fortran_COMPILER_ID

  This may be set to a string identifying a Fortran compiler vendor.
  See :variable:`Fortran_COMPILER_VENDORS`.

  If not already set in the including scope, the value is set to:

  * the ``<compiler_id>`` of :variable:`Fortran_<Fortran_COMPILER_ID>_EXECUTABLE`
    if defined. If multiple variable of the same form are passed, ``<compiler_id>``
    are tested in the order of :variable:`Fortran_COMPILER_VENDORS`.

  * or the :variable:`CMAKE_<LANG>_COMPILER_ID` for Fortran language if defined.

  * or the ``<compiler_id>`` of :variable:`CMAKE_<LANG>_COMPILER` for Fortran language.

  * or the ``<compiler_id>` of any available Fortran compiler. A logic
    similar to the one of :module:`CheckLanguage` is used.

.. variable:: Fortran_<Fortran_COMPILER_ID>_EXECUTABLE

  This variable may be set to explicitly select the <Fortran_COMPILER_ID>
  executable and bypass the discovery process. If the compile executable
  is not in the ``PATH``, this variable may be set to ensure it is
  discovered.


Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

.. variable:: Fortran_<Fortran_COMPILER_ID>_EXECUTABLE

  Path to the <Fortran_COMPILER_ID> executable.

.. variable:: Fortran_<Fortran_COMPILER_ID>_IMPLICIT_LINK_LIBRARIES
.. variable:: Fortran_<Fortran_COMPILER_ID>_IMPLICIT_LINK_DIRECTORIES
.. variable:: Fortran_<Fortran_COMPILER_ID>_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES

  List of implicit linking variables associated with ``<Fortran_COMPILER_ID>``.

  If the variables :variable:`CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES`,
  :variable:`CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES`
  and :variable:`CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES` are *NOT* already
  defined in the including project, they will be conveniently initialized by this module
  using the corresponding ``Fortran_<Fortran_COMPILER_ID>_IMPLICIT_LINK_*`` variables.

  Note that setting the ``CMAKE_Fortran_IMPLICIT_LINK_*`` variables ensures that the
  imported targets having the :variable:`IMPORTED_LINK_INTERFACE_LANGUAGES`
  property set to "Fortran" automatically link against the associated libraries.

.. variable:: Fortran_<Fortran_COMPILER_ID>_RUNTIME_LIBRARIES

  List of ``<Fortran_COMPILER_ID>`` runtime libraries.

  These libraries may be distributed along side the compiled binaries. This may be done
  by explicitly using :command:`install` or by setting the variable ``CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS``
  when using :module:`InstallRequiredSystemLibraries` module.

  Libraries expected to be available on most Unix systems (excluding macOS)
  are not listed. The list is derived from the `manylinux1 policy
  <https://www.python.org/dev/peps/pep-0513/#the-manylinux1-policy>`_. Libraries
  excluded are ``c``, ``crypt``, ``dl``, ``gcc_s``, ``m``, ``nsl``, ``rt``,
  ``util``, ``pthread`` and ``stdc++``.

.. variable:: Fortran_<Fortran_COMPILER_ID>_RUNTIME_DIRECTORIES

  List of directories corresponding to :variable:`Fortran_<Fortran_COMPILER_ID>_RUNTIME_LIBRARIES`.

  This list of directories may be used to configure a launcher.

.. variable:: Fortran_COMPILER_VENDORS

  List of short unique string identifying supported Fortran compiler vendors.


Functions
^^^^^^^^^

The module provides the following functions

.. command:: Fortran_InstallLibrary

  This function behaves like the :command:`install` command unless the filename
  is both a symlink and the resolved filename is of the form `libname.so.1[.2[.3]]`.
  In that case, a symlink for each version is installed: ``libname.so``,
  ``libname.so.1`` and ``libname.so.1.2``

  ::

    Fortran_InstallLibrary(
      FILES file1 ...
      DESTINATION <dir>
      [PERMISSIONS permissions...]
      [COMPONENT <component>]
      )

#]=======================================================================]

function(Fortran_InstallLibrary)
  set(options)
  set(oneValueArgs DESTINATION COMPONENT)
  set(multiValueArgs FILES PERMISSIONS)
  cmake_parse_arguments(_ffil "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(var DESTINATION FILES)
    if(NOT DEFINED _ffil_${var})
      message(FATAL_ERROR "Argument ${var} is required !")
    endif()
  endforeach()

  if(DEFINED _ffil_PERMISSIONS)
    set(install_permissions PERMISSIONS ${_ffil_PERMISSIONS})
  endif()
  foreach(_lib_path IN LISTS _ffil_FILES)
    get_filename_component(_resolved_lib_path ${_lib_path} REALPATH)
    get_filename_component(_resolved_lib_filename ${_resolved_lib_path} NAME)
    # Extract version
    install(FILES ${_resolved_lib_path} COMPONENT ${_ffil_COMPONENT} DESTINATION ${_ffil_DESTINATION} ${install_permissions})
    if(_resolved_lib_filename MATCHES "^(.+\.so)(\.[0-9]+)?(\.[0-9]+)?(\.[0-9]+)?$")
      set(_base ${CMAKE_MATCH_1})
      set(_x ${CMAKE_MATCH_2})
      set(_y ${CMAKE_MATCH_3})
      set(_link_names "${_base}")
      if(_x)
        list(APPEND _link_names "${_base}${_x}")
      endif()
      if(_y)
        list(APPEND _link_names "${_base}${_x}${_y}")
      endif()
      foreach(link IN LISTS _link_names)
        install(CODE "execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
\"\${CMAKE_INSTALL_PREFIX}/${_ffil_DESTINATION}/${_resolved_lib_filename}\" \"\${CMAKE_INSTALL_PREFIX}/${_ffil_DESTINATION}/${link}\"
)"
          COMPONENT ${_ffil_COMPONENT}
          )
      endforeach()
    endif()
  endforeach()
endfunction()

function(_fortran_assert)
  if(NOT (${ARGN}))
    message(FATAL_ERROR "assertion error [${ARGN}]")
  endif()
endfunction()

function(_fortran_msg)
  if(NOT Fortran_FIND_QUIETLY)
    message(STATUS "${ARGN}")
  endif()
endfunction()

function(_fortran_set_implicit_linking_cache_variables)
  # Caller must defined these variables
  _fortran_assert(DEFINED _id)
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES)

  _fortran_msg("Fortran_${_id}_IMPLICIT_LINK_LIBRARIES=${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES}")
  _fortran_msg("Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES=${Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES}")
  _fortran_msg("Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES=${Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}")

  set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES "${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES}" CACHE STRING "${_id} Fortran compiler implicit link libraries")
  mark_as_advanced(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)

  set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES "${Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES}" CACHE STRING "${_id} Fortran compiler implicit link directories")
  mark_as_advanced(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)

  set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "${Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}" CACHE STRING "${_id} Fortran compiler implicit link framework directories")
  mark_as_advanced(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES)
endfunction()

function(_fortran_retrieve_implicit_link_info)
  # Caller must defined these variables
  _fortran_assert(DEFINED _id)
  _fortran_assert(DEFINED Fortran_${_id}_EXECUTABLE)

  set(_additional_cmake_options ${ARGN})
  if(NOT DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
    set(_desc "Retrieving ${_id} Fortran compiler implicit link info")
    _fortran_msg(${_desc})

    if(NOT Fortran_${_id}_EXECUTABLE)
      _fortran_msg("${_desc} - failed")
      set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES NOTFOUND PARENT_SCOPE)
      set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES NOTFOUND PARENT_SCOPE)
      set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES NOTFOUND PARENT_SCOPE)
      return()
    endif()

    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckFortran${_id})
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckFortran${_id}/CMakeLists.txt"
      "cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CheckFortran${_id} Fortran)
file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/result.cmake\"
\"
set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES \\\"\${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}\\\")
set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES \\\"\${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}\\\")
set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES \\\"\${CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}\\\")
\")
")
    if(CMAKE_GENERATOR_INSTANCE)
      set(_D_CMAKE_GENERATOR_INSTANCE "-DCMAKE_GENERATOR_INSTANCE:INTERNAL=${CMAKE_GENERATOR_INSTANCE}")
    else()
      set(_D_CMAKE_GENERATOR_INSTANCE "")
    endif()
    execute_process(
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckFortran${_id}
      COMMAND ${CMAKE_COMMAND} . -DCMAKE_Fortran_COMPILER:FILEPATH=${Fortran_${_id}_EXECUTABLE}
                                 ${_additional_cmake_options}
                                 -G ${CMAKE_GENERATOR}
                                 -A "${CMAKE_GENERATOR_PLATFORM}"
                                 -T "${CMAKE_GENERATOR_TOOLSET}"
                                 ${_D_CMAKE_GENERATOR_INSTANCE}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE result
      )
    include(${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckFortran${_id}/result.cmake OPTIONAL)
    if(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES AND "${result}" STREQUAL "0")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "${_desc} passed with the following output:\n"
        "${output}\n")
      if(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
        list(REMOVE_DUPLICATES Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
      endif()
      if(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)
        list(REMOVE_DUPLICATES Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)
      endif()
      if(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES)
        list(REMOVE_DUPLICATES Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES)
      endif()
    else()
      set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES NOTFOUND)
      set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES NOTFOUND)
      set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES NOTFOUND)
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "${_desc} failed with the following output:\n"
        "${output}\n")
    endif()
    _fortran_msg("${_desc} - done")
  endif()

  set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES "${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES}" PARENT_SCOPE)
  set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES "${Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES}" PARENT_SCOPE)
  set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "${Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}" PARENT_SCOPE)
endfunction()

function(_fortran_set_runtime_cache_variables)
  # Caller must defined these variables
  _fortran_assert(DEFINED _id)
  _fortran_assert(DEFINED _link_libs)
  _fortran_assert(DEFINED _runtime_lib_dirs)
  _fortran_assert(DEFINED _runtime_lib_suffix)

  if(NOT DEFINED Fortran_${_id}_RUNTIME_LIBRARIES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_runtime_lib_suffix}")

    set(_runtime_libs)
    set(_runtime_dirs)
    foreach(_lib IN LISTS _link_libs)
      get_filename_component(_lib ${_lib} NAME_WE)
      find_library(
        Fortran_${_id}_${_lib}_RUNTIME_LIBRARY ${_lib}
        HINTS ${_runtime_lib_dirs} NO_DEFAULT_PATH
        )
      if(NOT Fortran_${_id}_${_lib}_RUNTIME_LIBRARY)
        unset(Fortran_${_id}_${_lib}_RUNTIME_LIBRARY CACHE) # Do not pollute the project cache
        continue()
      endif()

      list(APPEND _runtime_libs ${Fortran_${_id}_${_lib}_RUNTIME_LIBRARY})

      get_filename_component(_runtime_dir ${Fortran_${_id}_${_lib}_RUNTIME_LIBRARY} DIRECTORY)
      list(APPEND _runtime_dirs ${_runtime_dir})
    endforeach()
    if(_runtime_dirs)
      list(REMOVE_DUPLICATES _runtime_dirs)
    endif()

    set(Fortran_${_id}_RUNTIME_LIBRARIES ${_runtime_libs} CACHE FILEPATH "${_id} Fortran compiler runtime libraries")
    mark_as_advanced(Fortran_${_id}_RUNTIME_LIBRARIES)

    set(Fortran_${_id}_RUNTIME_DIRECTORIES ${_runtime_dirs} CACHE FILEPATH "${_id} Fortran compiler runtime directories")
    mark_as_advanced(Fortran_${_id}_RUNTIME_DIRECTORIES)

    _fortran_msg("Fortran_${_id}_RUNTIME_LIBRARIES=${Fortran_${_id}_RUNTIME_LIBRARIES}")
    _fortran_msg("Fortran_${_id}_RUNTIME_DIRECTORIES=${Fortran_${_id}_RUNTIME_DIRECTORIES}")
  endif()
endfunction()

set(Fortran_COMPILER_VENDORS GNU Intel Absoft PGI Flang PathScale XL VisualAge NAG G95 Cray SunPro)

# Vendor-specific compiler names (copied from CMakeDetermineFortranCompiler.cmake)
set(_Fortran_COMPILER_NAMES_GNU       gfortran gfortran-4 g95 g77 f95)
set(_Fortran_COMPILER_NAMES_Intel     ifort ifc efc)
set(_Fortran_COMPILER_NAMES_Absoft    af95 af90 af77)
set(_Fortran_COMPILER_NAMES_PGI       pgf95 pgfortran pgf90 pgf77)
set(_Fortran_COMPILER_NAMES_Flang     flang)
set(_Fortran_COMPILER_NAMES_PathScale pathf2003 pathf95 pathf90)
set(_Fortran_COMPILER_NAMES_XL        xlf)
set(_Fortran_COMPILER_NAMES_VisualAge xlf95 xlf90 xlf)
set(_Fortran_COMPILER_NAMES_NAG       nagfor)
set(_Fortran_COMPILER_NAMES_G95       g95)
set(_Fortran_COMPILER_NAMES_Cray      ftn)
set(_Fortran_COMPILER_NAMES_SunPro    f90 f77)

function(_fortran_find_vendor_compiler_executable _id)

  # Adapted from _cmake_find_compiler() available in CMakeDetermineCompiler.cmake

  # Use already-enabled languages for reference.
  get_property(_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM _languages "Fortran")

  # Compiler list
  set(_Fortran_COMPILER_LIST ${_Fortran_COMPILER_NAMES_${_id}})

  # Compiler directories
  set(_Fortran_COMPILER_HINTS)

  # Consider directory associated with FC env. variable
  if(DEFINED ENV{FC})
    get_filename_component(_hint "$ENV{FC}" DIRECTORY)
    list(APPEND _Fortran_COMPILER_HINTS "${_hint}")
  endif()

  # Look for directories containing compilers.
  foreach(l IN ITEMS ${_languages} Fortran)
    if(CMAKE_${l}_COMPILER AND IS_ABSOLUTE "${CMAKE_${l}_COMPILER}")
      get_filename_component(_hint "${CMAKE_${l}_COMPILER}" PATH)
      if(IS_DIRECTORY "${_hint}")
        list(APPEND _Fortran_COMPILER_HINTS "${_hint}")
      endif()
      unset(_hint)
    endif()
  endforeach()

  # Find the compiler.
  if(_Fortran_COMPILER_HINTS)
    # Prefer directories containing compilers of reference languages.
    list(REMOVE_DUPLICATES _Fortran_COMPILER_HINTS)
    find_program(Fortran_${_id}_EXECUTABLE
      NAMES ${_Fortran_COMPILER_LIST}
      PATHS ${_Fortran_COMPILER_HINTS}
      NO_DEFAULT_PATH
      DOC "${_id} Fortran compiler")
  endif()

  find_program(Fortran_${_id}_EXECUTABLE NAMES ${_Fortran_COMPILER_LIST} DOC "${_id} Fortran compiler")
endfunction()

function(_fortran_run_compiler_test test_id)
  set(build_dir ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CheckFortran_${test_id})
  set(additonal_cmake_options ${ARGN})
  file(REMOVE_RECURSE ${build_dir})
  file(WRITE "${build_dir}/CMakeLists.txt"
    "cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(CheckFortran Fortran)
file(WRITE \"\${CMAKE_CURRENT_BINARY_DIR}/result.cmake\" \"
  set(Fortran_COMPILER \\\"\${CMAKE_Fortran_COMPILER}\\\")
  set(Fortran_COMPILER_ID \\\"\${CMAKE_Fortran_COMPILER_ID}\\\")
\")
")
  if(CMAKE_GENERATOR_INSTANCE)
    set(_D_CMAKE_GENERATOR_INSTANCE "-DCMAKE_GENERATOR_INSTANCE:INTERNAL=${CMAKE_GENERATOR_INSTANCE}")
  else()
    set(_D_CMAKE_GENERATOR_INSTANCE "")
  endif()
  execute_process(
    WORKING_DIRECTORY ${build_dir}
    COMMAND ${CMAKE_COMMAND} . -G ${CMAKE_GENERATOR}
                               -A "${CMAKE_GENERATOR_PLATFORM}"
                               -T "${CMAKE_GENERATOR_TOOLSET}"
                               ${_D_CMAKE_GENERATOR_INSTANCE}
                               ${additonal_cmake_options}
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE result
    )
  include(${build_dir}/result.cmake OPTIONAL)
  set(_desc "Fortran compiler test ${test_id}")
  if(Fortran_COMPILER AND "${result}" STREQUAL "0")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "${_desc} passed with the following output:\n"
      "${output}\n")
    if(Fortran_COMPILER_ID STREQUAL "")
      set(Fortran_COMPILER_ID "Fortran_COMPILER_ID-NOTFOUND")
    endif()
  else()
    set(Fortran_COMPILER "Fortran_COMPILER-NOTFOUND")
    set(Fortran_COMPILER_ID "Fortran_COMPILER_ID-NOTFOUND")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "${_desc} failed with the following output:\n"
      "${output}\n")
  endif()
  set(Fortran_COMPILER ${Fortran_COMPILER} PARENT_SCOPE)
  set(Fortran_COMPILER_ID ${Fortran_COMPILER_ID} PARENT_SCOPE)
endfunction()

function(_fortran_find_any_compiler_executable)
  set(_desc "Looking up any Fortran compiler")
  _fortran_msg(${_desc})
  set(_details )
  if(EXISTS "$ENV{FC}")
    set(Fortran_COMPILER "$ENV{FC}")
    _fortran_msg("${_desc} - ${Fortran_COMPILER} (value of FC environment variable)")
    _fortran_get_compiler_id($ENV{FC})
  else()
    _fortran_run_compiler_test("any")
    _fortran_msg("${_desc} - ${Fortran_COMPILER}")
  endif()
  set(Fortran_COMPILER_ID ${Fortran_COMPILER_ID} PARENT_SCOPE)
  if(Fortran_COMPILER_ID)
    set(Fortran_${Fortran_COMPILER_ID}_EXECUTABLE ${Fortran_COMPILER} PARENT_SCOPE)
  endif()
endfunction()

function(_fortran_get_compiler_id compiler_executable)
  set(_desc "Extracting compiler identifier for ${compiler_executable}")
  _fortran_msg("${_desc}")

  # Get executable name
  get_filename_component(compiler_name ${compiler_executable} NAME_WE)

  # Check in list of known compiler names
  foreach(possible_id IN LISTS Fortran_COMPILER_VENDORS)
    foreach(possible_compiler_name IN LISTS _Fortran_COMPILER_NAMES_${possible_id})
      if(compiler_name STREQUAL ${possible_compiler_name})
        set(Fortran_COMPILER_ID ${possible_id})
        _fortran_msg("${_desc} - ${Fortran_COMPILER_ID}")
        set(Fortran_COMPILER_ID ${Fortran_COMPILER_ID} PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()

  # Resolve symlink
  #get_filename_component(compiler_executable ${compiler_executable} REALPATH) # Should we do this ?
  # Get unique identifier for build caching
  string(SHA256 compiler_path_hash ${compiler_executable})
  string(SUBSTRING ${compiler_path_hash} 0 7 compiler_build_cache_id)
  set(compiler_build_cache_id "${compiler_name}_${compiler_build_cache_id}")
  # Attempt to retrieve compiler id
  _fortran_run_compiler_test(${compiler_build_cache_id}
    -DCMAKE_Fortran_COMPILER:FILEPATH=${compiler_executable}
    )
  _fortran_msg("${_desc} - ${Fortran_COMPILER_ID}")
  set(Fortran_COMPILER_ID ${Fortran_COMPILER_ID} PARENT_SCOPE)
endfunction()

function(_fortran_set_unix_runtime_cache_variables)
  # Caller must defined these variables
   _fortran_assert(DEFINED _id)
   _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
   _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)

  set(_link_libs ${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES})
  if(UNIX AND NOT APPLE)
    # These libraries are expected to be available.
    # See https://www.python.org/dev/peps/pep-0513/#the-manylinux1-policy
    list(REMOVE_ITEM _link_libs c crypt dl gcc_s m nsl rt util pthread stdc++)
  endif()
  set(_runtime_lib_dirs ${Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES})
  set(_runtime_lib_suffix ".so")
  if(APPLE)
    set(_runtime_lib_suffix ".dylib")
  endif()
  _fortran_set_runtime_cache_variables()
endfunction()

# Check for Fortran_<compile_id>_EXECUTABLE variable
if(NOT DEFINED Fortran_COMPILER_ID)
  set(_msg "Checking if Fortran_<compile_id>_EXECUTABLE is defined")
  _fortran_msg("${_msg}")
  foreach(_possible_id IN LISTS Fortran_COMPILER_VENDORS)
    if(Fortran_${_possible_id}_EXECUTABLE)
      set(Fortran_COMPILER_ID ${_possible_id})
      _fortran_msg("${_msg} - yes [compiler_id:${Fortran_COMPILER_ID};compiler:${Fortran_${Fortran_COMPILER_ID}_EXECUTABLE}]")
      break()
    endif()
  endforeach()
  if(NOT DEFINED Fortran_COMPILER_ID)
    _fortran_msg("${_msg} - no")
  endif()
endif()

# Check for enabled Fortran language
if(NOT DEFINED Fortran_COMPILER_ID)
  set(_msg "Checking if CMAKE_Fortran_COMPILER_ID is defined")
  _fortran_msg("${_msg}")
  if(CMAKE_Fortran_COMPILER_ID)
    set(Fortran_COMPILER_ID ${CMAKE_Fortran_COMPILER_ID})
    _fortran_msg("${_msg} - ${Fortran_COMPILER_ID}")
  else()
    _fortran_msg("${_msg} - no")
  endif()

  # If a CMAKE_Fortran_COMPILER is provided without CMAKE_Fortran_COMPILER_ID, let's retrieve it
  if(CMAKE_Fortran_COMPILER AND NOT CMAKE_Fortran_COMPILER_ID)
    #_fortran_msg("CMAKE_Fortran_COMPILER is set without CMAKE_Fortran_COMPILER_ID")
    _fortran_get_compiler_id(${CMAKE_Fortran_COMPILER}) # Set Fortran_COMPILER_ID in current scope
  endif()
endif()

# Attempt to find any fortran compiler
if(NOT DEFINED Fortran_COMPILER_ID)
  _fortran_find_any_compiler_executable()
endif()

if(Fortran_COMPILER_ID IN_LIST Fortran_COMPILER_VENDORS AND NOT DEFINED Fortran_${Fortran_COMPILER_ID}_EXECUTABLE)
  set(_msg "Looking up ${Fortran_COMPILER_ID} Fortran compiler")
  _fortran_msg("${_msg}")
  _fortran_find_vendor_compiler_executable(${Fortran_COMPILER_ID})
  if(Fortran_${Fortran_COMPILER_ID}_EXECUTABLE)
    _fortran_msg("${_msg} - ${Fortran_${Fortran_COMPILER_ID}_EXECUTABLE}")
  else()
    _fortran_msg("${_msg} - not found")
  endif()
endif()

# convenient shorter variable name
set(_id ${Fortran_COMPILER_ID})

set(_additional_required_vars)

if(_id STREQUAL "Flang")
  get_filename_component(_flang_bin_dir ${Fortran_${_id}_EXECUTABLE} DIRECTORY)
  if(CMAKE_HOST_WIN32)
    # Set companion compiler variables
    find_program(Fortran_${_id}_CLANG_CL_EXECUTABLE clang-cl.exe HINTS ${_flang_bin_dir})
    list(APPEND _additional_required_vars Fortran_${_id}_CLANG_CL_EXECUTABLE)

    # Set implicit linking variables
    if(NOT DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
      set(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES flangmain flang flangrti ompstub)
      set(Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES ${_flang_bin_dir}/../lib)
      set(Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
      _fortran_set_implicit_linking_cache_variables()
    endif()

    # Set runtime variables
    set(_link_libs ${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES})
    
    # Explicitly add libomp for Flang build runtime libs
    list(APPEND _link_libs libomp)
    
    set(_runtime_lib_dirs ${_flang_bin_dir})
    set(_runtime_lib_suffix ".dll")
    _fortran_set_runtime_cache_variables()

  else()
    # Set implicit linking variables
    if(NOT DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
      _fortran_retrieve_implicit_link_info(
        -DCMAKE_EXE_LINKER_FLAGS:STRINGS=-lomp # This is a dependency of libflang and libflangrti
        -DCMAKE_POLICY_DEFAULT_CMP0056:STRING=NEW
        )
      _fortran_set_implicit_linking_cache_variables()
    endif()

    # Set runtime variables
    _fortran_set_unix_runtime_cache_variables()

  endif()
  unset(_flang_bin_dir)

elseif(_id MATCHES "^GNU|G95|Intel|SunPro|Cray|G95|PathScale|Absoft|XL|VisualAge|PGI|NAG$")

  # Set implicit linking variables
  if(NOT DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
    _fortran_retrieve_implicit_link_info()
    _fortran_set_implicit_linking_cache_variables()
  endif()

  # Set runtime variables
  _fortran_set_unix_runtime_cache_variables()

elseif(_id MATCHES "^zOS|HP$")
  message(FATAL_ERROR "Setting Fortran_COMPILER_ID to '${_id}' is not yet supported")
endif()

if(_id)
  list(APPEND _additional_required_vars Fortran_${_id}_EXECUTABLE)

  # all variables must be defined
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)
  _fortran_assert(DEFINED Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES)
  _fortran_assert(DEFINED Fortran_${_id}_RUNTIME_LIBRARIES)
  _fortran_assert(DEFINED Fortran_${_id}_RUNTIME_DIRECTORIES)

  # directory variable is required if corresponding library variable is non-empty
  if(Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
    list(APPEND _additional_required_vars Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES)
  endif()
  if(Fortran_${_id}_RUNTIME_LIBRARIES)
    list(APPEND _additional_required_vars Fortran_${_id}_RUNTIME_DIRECTORIES)
  endif()
endif()

# outputs
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Fortran
  REQUIRED_VARS
    Fortran_COMPILER_ID
    ${_additional_required_vars}
  )

# conveniently set CMake implicit linking variables it not already defined
if(NOT DEFINED CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES
    AND NOT DEFINED CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES
    AND NOT DEFINED CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES
    AND _id
    AND Fortran_${_id}_IMPLICIT_LINK_LIBRARIES)
  set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES ${Fortran_${_id}_IMPLICIT_LINK_LIBRARIES})
  set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES ${Fortran_${_id}_IMPLICIT_LINK_DIRECTORIES})
  set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES ${Fortran_${_id}_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES})
endif()

# clean
unset(_additional_required_vars)
unset(_find_compiler_hints)
unset(_id)
