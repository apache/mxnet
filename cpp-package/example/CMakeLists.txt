if(NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif()

set(CPP_EXAMPLE_LIBS
  ${BEGIN_WHOLE_ARCHIVE} mxnet ${END_WHOLE_ARCHIVE}
  ${mxnet_LINKER_LIBS}
  )

set(CPP_PACKAGE_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/mxnet-cpp/)

set(CPPEX_DEPS cpp_package_op_h)

file(GLOB_RECURSE CPP_PACKAGE_HEADERS
  "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
  "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
  )

add_custom_target(
  cpp_package_deploy_library ALL
  DEPENDS mxnet
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mxnet> $<TARGET_FILE_DIR:mlp>
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(lenet lenet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(lenet ${CPP_EXAMPLE_LIBS})
add_dependencies(lenet ${CPPEX_DEPS})

add_executable(lenet_with_mxdataiter lenet_with_mxdataiter.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(lenet_with_mxdataiter ${CPP_EXAMPLE_LIBS})
add_dependencies(lenet_with_mxdataiter ${CPPEX_DEPS})

add_executable(alexnet alexnet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(alexnet ${CPP_EXAMPLE_LIBS})
add_dependencies(alexnet ${CPPEX_DEPS})

add_executable(charRNN charRNN.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(charRNN ${CPP_EXAMPLE_LIBS})
add_dependencies(charRNN ${CPPEX_DEPS})

add_executable(googlenet googlenet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(googlenet ${CPP_EXAMPLE_LIBS})
add_dependencies(googlenet ${CPPEX_DEPS})

add_executable(inception_bn inception_bn.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(inception_bn ${CPP_EXAMPLE_LIBS})
add_dependencies(inception_bn ${CPPEX_DEPS})

add_executable(mlp mlp.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(mlp ${CPP_EXAMPLE_LIBS})
add_dependencies(mlp ${CPPEX_DEPS})

add_executable(mlp_cpu mlp_cpu.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(mlp_cpu ${CPP_EXAMPLE_LIBS})
add_dependencies(mlp_cpu ${CPPEX_DEPS})

add_executable(mlp_gpu mlp_gpu.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(mlp_gpu ${CPP_EXAMPLE_LIBS})
add_dependencies(mlp_gpu ${CPPEX_DEPS})

add_executable(resnet resnet.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(resnet ${CPP_EXAMPLE_LIBS})
add_dependencies(resnet ${CPPEX_DEPS})
