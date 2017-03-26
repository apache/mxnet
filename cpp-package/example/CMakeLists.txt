
if(NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif()

set(CPP_EXAMPLE_LIBS
  rt
  mxnet
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(lenet lenet.cpp)
target_link_libraries(lenet ${CPP_EXAMPLE_LIBS})

add_executable(lenet_with_mxdataiter lenet.cpp)
target_link_libraries(lenet_with_mxdataiter ${CPP_EXAMPLE_LIBS})

add_executable(alexnet alexnet.cpp)
target_link_libraries(alexnet ${CPP_EXAMPLE_LIBS})

add_executable(charRNN charRNN.cpp)
target_link_libraries(charRNN ${CPP_EXAMPLE_LIBS})

add_executable(googlenet googlenet.cpp)
target_link_libraries(googlenet ${CPP_EXAMPLE_LIBS})

add_executable(inception_bn inception_bn.cpp)
target_link_libraries(inception_bn ${CPP_EXAMPLE_LIBS})

add_executable(mlp mlp.cpp)
target_link_libraries(mlp ${CPP_EXAMPLE_LIBS})

add_executable(resnet resnet.cpp)
target_link_libraries(resnet ${CPP_EXAMPLE_LIBS})
