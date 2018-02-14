cmake_minimum_required(VERSION 3.0.2)

if (NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif ()

project(sharedmemory_helper)

file(GLOB_RECURSE SHARED_MEMORY_HELPER_SOURCE
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
file(GLOB_RECURSE MXNET_HEADERS
     "${CMAKE_CURRENT_SOURCE_DIR}/../../include/*.h")
file(GLOB_RECURSE MXNET_PRIVATE_HEADERS
     "${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.h")

add_executable(sharedmemory_helper
               ${SHARED_MEMORY_HELPER_SOURCE}
               ${MXNET_HEADERS}
               ${MXNET_PRIVATE_HEADERS})

if (UNITTEST_STATIC_LINK)
  target_link_libraries(sharedmemory_helper
                        ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
                        dmlc
                        ${mxnet_LINKER_LIBS}
                        ${pslite_LINKER_LIBS}
                        )
else ()
  target_link_libraries(sharedmemory_helper
                        dmlc
                        ${nnvm_LINKER_LIBS}
                        ${mxnet_LINKER_LIBS}
                        mxnet
                        ${pslite_LINKER_LIBS}
                        )
endif ()
