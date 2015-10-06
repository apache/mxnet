cmake_minimum_required(VERSION 2.8.7)

project(mxnet C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
include(mshadow/cmake/mshadow.cmake)
include(mshadow/cmake/Utils.cmake)
include(mshadow/cmake/Cuda.cmake)

mxnet_option(USE_OPENCV "Build with OpenCV support" ON)
mxnet_option(USE_OPENMP "Build with Openmp support" ON)
mxnet_option(USE_CUDNN "Build with cudnn support" ON)

include_directories("include")
include_directories("mshadow")
include_directories("dmlc-core/include")

add_subdirectory("dmlc-core")


add_definitions(-DDMLC_USE_CXX11)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DMXNET_EXPORTS)


if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
  list(APPEND mshadow_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  add_definitions(-DMXNET_USE_OPENCV=1)
endif()

if(USE_OPENMP)
 FIND_PACKAGE( OpenMP REQUIRED)  
 if(OPENMP_FOUND)  
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
 endif()  
endif()

# cudnn detection
if(USE_CUDNN)
  detect_cuDNN()
  if(HAVE_CUDNN)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND mshadow_LINKER_LIBS ${CUDNN_LIBRARY})
	add_definitions(-DMSHADOW_USE_CUDNN=1)
  endif()
endif()

mxnet_source_group("Source"   GLOB_RECURSE "src/*.cc")
mxnet_source_group("Source\\Cuda" GLOB_RECURSE "src/*.cu")


FILE(GLOB_RECURSE SOURCE "src/*.cc")
FILE(GLOB_RECURSE cuda "src/*.cu")

if(USE_CUDA)
  # define preprocessor macro so that we will not include the generated forcelink header
  mshadow_cuda_compile(cuda_objs ${cuda})
  list(APPEND SOURCE ${cuda_objs} ${cuda})
endif()

add_library(mxnet SHARED ${SOURCE}) 
target_link_libraries(mxnet ${mshadow_LINKER_LIBS})
target_link_libraries(mxnet dmlccore)


