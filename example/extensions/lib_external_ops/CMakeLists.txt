# specify CXX sources
FILE(GLOB CXX_SRCS
  # Required files
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/lib_api.cc
  # Your custom files
  ${CMAKE_CURRENT_SOURCE_DIR}/init_lib.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/min_ex.cc
  )

# create library & set libraries
add_library(external_lib SHARED ${CXX_SRCS})
target_link_libraries(external_lib PUBLIC mxnet)

if(USE_CUDA)
  # specify GPU sources (optional)
  FILE(GLOB CU_SRCS "*.cu")
  target_sources(external_lib PUBLIC ${CU_SRCS})
endif(USE_CUDA)

if(UNIX)
  # unix-specific build configuration
  if(USE_CUDA)
    # unix+GPU-specific build configuration
  endif(USE_CUDA)
endif()

if(MSVC)
  message(FATAL_ERROR "Windows builds are not support for external ops")
endif()
