cmake_minimum_required(VERSION 2.8.7)

project(mshadow C CXX)

set(mshadow_LINT_DIRS mshadow mshadow-ps)
add_custom_target(mshadow_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DLINT_DIRS=${mshadow_LINT_DIRS} -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DPROJECT_NAME=mshadow -P ${PROJECT_SOURCE_DIR}/../dmlc-core/cmake/lint.cmake)
