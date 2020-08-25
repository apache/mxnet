# specify CXX sources
FILE(GLOB CXX_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/min_ex.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/init_lib.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/lib_api.cc)

# specify GPU sources (optional)
FILE(GLOB CU_SRCS "*.cu")

# create library & set libraries
add_library(external_lib SHARED ${CXX_SRCS})
target_link_libraries(external_lib PUBLIC mshadow mxnet)

# generic GPU stuff
if(USE_CUDA)
  target_sources(external_lib $CU_SRCS)
endif(USE_CUDA)

if(UNIX)
  # unix-specific stuff
  if(USE_CUDA)
    # unix+GPU-specific stuff
    target_compile_options(external_lib PUBLIC -shared)
  endif(USE_CUDA)
elseif(MSVC)
  # windows-specific stuff
  target_compile_options(external_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/LD>")
  target_compile_options(external_lib PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MT>")
  set_target_properties(external_lib PROPERTIES PREFIX "lib")
  if(USE_CUDA)
    # windows+GPU-specific stuff
    target_compile_options(external_lib PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-LD -MT>")
  endif(USE_CUDA)
endif()
