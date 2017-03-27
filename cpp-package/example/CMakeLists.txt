
if(NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif()

set(CPP_EXAMPLE_LIBS
  rt
  mxnet
  )

set(CPP_PACKAGE_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/mxnet-cpp/)

file(GLOB_RECURSE CPP_PACKAGE_HEADERS
  "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
  "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(lenet lenet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(lenet ${CPP_EXAMPLE_LIBS})
add_dependencies(lenet cpp_package_op_h)

add_executable(lenet_with_mxdataiter lenet_with_mxdataiter.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(lenet_with_mxdataiter ${CPP_EXAMPLE_LIBS})
add_dependencies(lenet_with_mxdataiter cpp_package_op_h)

add_executable(alexnet alexnet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(alexnet ${CPP_EXAMPLE_LIBS})
add_dependencies(alexnet cpp_package_op_h)

add_executable(charRNN charRNN.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(charRNN ${CPP_EXAMPLE_LIBS})
add_dependencies(charRNN cpp_package_op_h)

add_executable(googlenet googlenet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(googlenet ${CPP_EXAMPLE_LIBS})
add_dependencies(googlenet cpp_package_op_h)

add_executable(inception_bn inception_bn.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(inception_bn ${CPP_EXAMPLE_LIBS})
add_dependencies(inception_bn cpp_package_op_h)

add_executable(mlp mlp.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(mlp ${CPP_EXAMPLE_LIBS})
add_dependencies(mlp cpp_package_op_h)

add_executable(resnet resnet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(resnet ${CPP_EXAMPLE_LIBS})
add_dependencies(resnet cpp_package_op_h)
