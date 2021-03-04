#===============================================================================
# Copyright 2016-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB HEADERS_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB HEADERS_SUBDIR
    ${PROJECT_BINARY_DIR}/include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/oneapi/dnnl/*.hpp
    )
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions(-DDNNL_DLL_EXPORTS)
endif()

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

if(NOT DNNL_VERBOSE)
    add_definitions(-DDISABLE_VERBOSE)
endif()

if(DNNL_ENABLE_CONCURRENT_EXEC)
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC)
endif()

if(DNNL_ENABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is enabled")
else()
    add_definitions(-DDNNL_DISABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is disabled")
endif()

if(DNNL_ENABLE_JIT_PROFILING AND UNIX AND NOT APPLE)
    # Not every compiler adds -ldl automatically
    list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
endif()

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
endif()

# Windows does not support weak/strong symbols and no guarrantees by the linker
# for out_of_memory testing to work. Not tested on macOS
if(UNIX)
    if(DNNL_ENABLE_MEM_DEBUG)
        add_definitions(-DDNNL_ENABLE_MEM_DEBUG)
    endif()
endif()

add_subdirectory(common)
add_subdirectory(cpu)

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(gpu)
endif()

if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
endif()

get_property(LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)
add_library(${LIB_NAME} ${DNNL_LIBRARY_TYPE}
    ${VERSION_RESOURCE_FILE} ${HEADERS_ROOT} ${HEADERS_SUBDIR} ${LIB_DEPS})

set_property(TARGET ${LIB_NAME} PROPERTY VERSION "${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
set_property(TARGET ${LIB_NAME} PROPERTY SOVERSION "${DNNL_VERSION_MAJOR}")

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    # $<INSTALL_PREFIX> is required for compatibility with cmake 2.8
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries_build(${LIB_NAME}
    "${EXTRA_SHARED_LIBS};${EXTRA_STATIC_LIBS}")
target_link_libraries_install(${LIB_NAME} "${EXTRA_SHARED_LIBS}")
if(DNNL_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${LIB_NAME} "${EXTRA_STATIC_LIBS}")
endif()

set(LIB_EXPORT_NAME "${LIB_NAME}-targets")
install(TARGETS ${LIB_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# If only Cmake could preserve the directory hierarchy...
foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()
foreach(header ${HEADERS_SUBDIR})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/oneapi/dnnl/")
endforeach()

# Write version and package config files
set(LIB_CONFIG_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}")
set(LIB_VERSION_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_NAME}-config-version.cmake")
set(LIB_CONFIG_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${LIB_NAME}-config.cmake")
write_basic_package_version_file(
    "${LIB_VERSION_FILE}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "../cmake/config.cmake.in"
    "${LIB_CONFIG_FILE}"
    INSTALL_DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})
string(TOUPPER "${LIB_NAME}::" LIB_NAMESPACE)
install(EXPORT ${LIB_EXPORT_NAME}
    NAMESPACE ${LIB_NAMESPACE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})

# Intel MKL-DNN compat cmake files
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND}
    -DDIR=\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake
    -P ${PROJECT_SOURCE_DIR}/cmake/gen_mkldnn_compat_cmakes.cmake)")

# Intel MKL-DNN compat libraries
if (WIN32)
    if (NOT MINGW)
        set(prefix "")
        set(ext ".lib")
    else()
        set(prefix "lib")
        if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
            set(ext ".dll.a")
        else()
            set(ext ".a")
        endif()
    endif()
    add_custom_target(compat_libs ALL
        ${CMAKE_COMMAND} -E copy
        $<TARGET_LINKER_FILE_DIR:${LIB_NAME}>/$<TARGET_LINKER_FILE_NAME:${LIB_NAME}>
        $<TARGET_LINKER_FILE_DIR:${LIB_NAME}>/${prefix}mkldnn${ext}
        # Workaround for MSB8065 warning.
        COMMAND ${CMAKE_COMMAND} -E touch "CMakeFiles/compat_libs"
        DEPENDS ${LIB_NAME})
    install(FILES $<TARGET_LINKER_FILE_DIR:${LIB_NAME}>/${prefix}mkldnn${ext}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
else()
    if(DNNL_LIBRARY_TYPE STREQUAL "SHARED")
        set_ternary(ext APPLE ".dylib" ".so")
        set(vers ".${DNNL_VERSION_MAJOR};.${DNNL_VERSION_MAJOR}.${DNNL_VERSION_MINOR}")
    else()
        set(ext ".a")
        set(vers "")
    endif()
    foreach(ver "" ${vers})
        set_ternary(ext_and_ver APPLE "${ver}${ext}" "${ext}${ver}")
        get_property(lib_location TARGET ${LIB_NAME} PROPERTY LIBRARY_OUTPUT_DIRECTORY)
        if(lib_location)
            set(compat_link "${lib_location}/libmkldnn${ext_and_ver}")
        else()
            set(compat_link "${CMAKE_CURRENT_BINARY_DIR}/libmkldnn${ext_and_ver}")
        endif()
        add_custom_command(OUTPUT ${compat_link}
            # to make the next command work fine
            COMMAND ${CMAKE_COMMAND} -E remove -f ${compat_link}
            COMMAND ${CMAKE_COMMAND} -E create_symlink libdnnl${ext_and_ver} ${compat_link}
            DEPENDS ${LIB_NAME})
        add_custom_target(compat_libs${ver} ALL DEPENDS ${compat_link})
        install(FILES ${compat_link} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    endforeach()
endif()

# Install custom find modules for transitive dependencies
set(LIB_CONFIG_INSTALL_DIR_COMPAT "${CMAKE_INSTALL_LIBDIR}/cmake/mkldnn")
if(DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    if(WIN32)
        install(FILES "../cmake/win/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(APPLE)
        install(FILES "../cmake/mac/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(UNIX)
        install(FILES "../cmake/lnx/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    endif()
    install(FILES "${CMAKE_INSTALL_PREFIX}/${LIB_CONFIG_INSTALL_DIR}/FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR_COMPAT})
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    install(FILES
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    install(FILES "../cmake/FindOpenCL.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR_COMPAT})
endif()

if(DNNL_SYCL_RUNTIME STREQUAL "COMPUTECPP")
    install(FILES
        "../cmake/FindComputeCpp.cmake"
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
elseif(DNNL_SYCL_RUNTIME STREQUAL "DPCPP")
    install(FILES
        "../cmake/FindLevelZero.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

# On Windows we need to add dnnl.dll path to CTESTCONFIG_PATH which is later
# passed to ctest and Visual Studio solutions
if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
            append_to_windows_path_list(CTESTCONFIG_PATH
                "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_TYPE}")
        endforeach()
    else()
        append_to_windows_path_list(CTESTCONFIG_PATH
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
endif()
