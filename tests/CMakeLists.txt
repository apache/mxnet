
# ---[ Google Test
if(NOT GTEST_ROOT)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/googletest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gtest")
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  else()
    enable_testing()
    find_package(GTest)
  endif()
endif()

if(GTEST_FOUND)

  enable_testing()

  file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/*.cc" "cpp/*.h")

  link_directories(${C_CXX_LINK_DIRECTORIES})
  add_executable(${PROJECT_NAME}_unit_tests ${UNIT_TEST_SOURCE})
  target_link_libraries(${PROJECT_NAME}_unit_tests
    ${mxnet_LINKER_LIBS}
    gtest
    mxnet
    )

  add_test(AllTestsIn${PROJECT_NAME}UnitTests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_unit_tests)

  add_custom_target(check
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_unit_tests
    DEPENDS ${PROJECT_NAME}_unit_tests
    COMMENT "Unit tests"
  )
endif(GTEST_FOUND)

