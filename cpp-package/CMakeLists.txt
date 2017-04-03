
if(USE_CPP_PACKAGE AND NOT MSVC)

set(CPP_PACKAGE_OP_H_HEADER ${CMAKE_CURRENT_LIST_DIR}/include/mxnet-cpp/op.h)

add_custom_target(
  cpp_package_op_h ALL
  BYPRODUCTS ${CPP_PACKAGE_OP_H_HEADER}
  MAIN_DEPENDENCY mxnet
  DEPENDS mxnet ${CMAKE_SOURCE_DIR}/cpp-package/src/OpWrapperGenerator/OpWrapperGenerator.py
  COMMAND echo "Running: OpWrapperGenerator.py"
  COMMAND python OpWrapperGenerator.py ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<TARGET_FILE:mxnet>
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cpp-package/src/OpWrapperGenerator/
)

add_subdirectory(example)

endif()