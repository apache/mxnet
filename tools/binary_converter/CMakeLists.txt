if(NOT MSVC AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(UNITTEST_STATIC_LINK ON)
endif()

add_executable(model-converter main.cpp)
include_directories(../../lib/)

if(UNITTEST_STATIC_LINK)
    target_link_libraries(model-converter
            ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
            dmlc
            ${mxnet_LINKER_LIBS}
            )
else()
    target_link_libraries(model-converter
            dmlc
            ${nnvm_LINKER_LIBS}
            ${mxnet_LINKER_LIBS}
            mxnet
            )
endif()
