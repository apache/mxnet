
# ---[ Google Test
if(NOT GTEST_ROOT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
    set(GTEST_FOUND ON)
  else()
    find_package(GTest)
  endif()
endif()

if(GTEST_FOUND)

  enable_testing()

  file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/*.cc")

  add_executable(${PROJECT_NAME}_unit_tests ${UNIT_TEST_SOURCE})

  target_link_libraries(${PROJECT_NAME}_unit_tests
    ${GTEST_LIBRARY}
    rt
    mxnet
    ${mxnet_LINKER_LIBS}
    )

  add_test(AllTestsIn${PROJECT_NAME}UnitTests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_unit_tests)

endif()

