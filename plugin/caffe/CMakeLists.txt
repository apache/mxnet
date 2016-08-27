cmake_minimum_required(VERSION 2.8.7)
set(CAFFE_ROOT ${PROJECT_SOURCE_DIR}/caffe)
set(CAFFE_ROOT_SRC ${CAFFE_ROOT}/src/caffe)



project(caffe C CXX)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

include(${CAFFE_ROOT}/cmake/Utils.cmake)
include(${CAFFE_ROOT}/cmake/Targets.cmake)
include(${CAFFE_ROOT}/cmake/Misc.cmake)
include(${CAFFE_ROOT}/cmake/Summary.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ConfigGen.cmake)
include(${CAFFE_ROOT}/cmake/ProtoBuf.cmake)
include(${CAFFE_ROOT}/cmake/Cuda.cmake)

if(NOT (USE_CUDNN AND USE_CUDA))
	add_definitions(-DCPU_ONLY)
endif()
if(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
	add_definitions(-DUSE_MKL)
endif()
add_definitions(-DCLOSE_HDF5)
add_definitions(-DCLOSE_CAFFE_TEST)
# add_definitions(-DGOOGLE_GLOG_DLL_DECL)

set(Caffe_INCLUDE_DIR "")
#set(Caffe_INCLUDE_DIR "")
set(Caffe_LINKER_LIBS "")

list(APPEND CMAKE_MODULE_PATH ${CAFFE_ROOT}/cmake/Modules)

# ---[ Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
list(APPEND Caffe_INCLUDE_DIR ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

find_package(Glog REQUIRED)
list(APPEND Caffe_INCLUDE_DIR ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(GFlags REQUIRED)
list(APPEND Caffe_INCLUDE_DIR ${GFLAGS_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARY})

list(APPEND Caffe_LINKER_LIBS "shlwapi")
	
# ---[ Config generation
configure_file(${CAFFE_ROOT}/cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
list(APPEND Caffe_INCLUDE_DIR ${CAFFE_ROOT}/include)
include_directories(SYSTEM ${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})


# generate protobuf sources
file(GLOB proto_files ${CAFFE_ROOT_SRC}/proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
list(APPEND Caffe_LINKER_LIBS proto) # note, crucial to prepend!

caffe_default_properties(proto)

# --[ Caffe library

# creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
caffe_pickup_caffe_sources(${CAFFE_ROOT})

if(HAVE_CUDA)
  caffe_cuda_compile(cuda_objs ${cuda})
  list(APPEND srcs ${cuda_objs} ${cuda})
endif()

add_library(caffe ${srcs})
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
caffe_default_properties(caffe)
set_target_properties(caffe PROPERTIES
    VERSION   ${CAFFE_TARGET_VERSION}
    SOVERSION ${CAFFE_TARGET_SOVERSION}
    )
	
get_directory_property(Caffe_INCLUDE_DIR INCLUDE_DIRECTORIES)	
set(Caffe_INCLUDE_DIR ${Caffe_INCLUDE_DIR} PARENT_SCOPE)

# ---[ Configuration summary
#caffe_print_configuration_summary()

# ---[ Export configs generation
#caffe_generate_export_configs()

