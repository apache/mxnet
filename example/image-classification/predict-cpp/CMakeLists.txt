# Check OpenCV
if(NOT USE_OPENCV OR NOT OpenCV_FOUND OR OpenCV_VERSION_MAJOR LESS 3)
  message(WARNING "\
OpenCV version >= 3 should be enabled and found to build image classification example, skipping...")
  return()
endif()

add_executable(image-classification-predict image-classification-predict.cc)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

target_link_libraries(image-classification-predict
                      dmlc
                      ${nnvm_LINKER_LIBS}
                      ${mxnet_LINKER_LIBS}
                      mxnet
                      )
add_dependencies(image-classification-predict mxnet)


