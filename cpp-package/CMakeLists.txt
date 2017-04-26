if(USE_CPP_PACKAGE)

set(CPP_PACKAGE_OP_H_HEADER ${CMAKE_CURRENT_LIST_DIR}/include/mxnet-cpp/op.h)

add_custom_target(
  cpp_package_op_h ALL
  BYPRODUCTS ${CPP_PACKAGE_OP_H_HEADER}
  MAIN_DEPENDENCY mxnet
  DEPENDS mxnet ${CMAKE_SOURCE_DIR}/cpp-package/scripts/OpWrapperGenerator.py
  COMMAND echo "Running: OpWrapperGenerator.py $<TARGET_FILE:mxnet>"
  COMMAND python OpWrapperGenerator.py $<TARGET_FILE:mxnet>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cpp-package/scripts/
)

add_subdirectory(example)

endif()