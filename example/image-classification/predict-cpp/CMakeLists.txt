# ---[ OpenCV
if(NOT USE_OPENCV OR NOT OpenCV_FOUND)
  message(WARNING "\
OpenCV should be enabled and found to build image classification example, skipping...")
  return()
endif()

add_executable(image-classification-predict image-classification-predict.cc)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

if(NOT MSVC)
  target_link_libraries(image-classification-predict
                        ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
                        dmlc
                        ${mxnet_LINKER_LIBS}
                        )
  add_dependencies(image-classification-predict mxnet_static)
else()
  target_link_libraries(image-classification-predict
                        dmlc
                        ${nnvm_LINKER_LIBS}
                        ${mxnet_LINKER_LIBS}
                        mxnet
                        )
  add_dependencies(image-classification-predict mxnet)
endif()


