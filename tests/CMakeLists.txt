
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

if(NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif()

# FIXME MSVC unit test linking issue
if(GTEST_FOUND AND NOT MSVC)
  enable_testing()

  file(GLOB_RECURSE UNIT_TEST_SOURCE "cpp/*.cc" "cpp/*.h")

  include_directories(cpp/include)

  if (NOT PRIVATE_RUNTIME_DIR)
   set(PRIVATE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()

  add_executable(${PROJECT_NAME}_unit_tests ${UNIT_TEST_SOURCE})
  set_property(TARGET ${PROJECT_NAME}_unit_tests
               PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PRIVATE_RUNTIME_DIR})

  if(UNITTEST_STATIC_LINK)
    target_link_libraries(${PROJECT_NAME}_unit_tests
      ${GTEST_LIBRARY}
      ${BEGIN_WHOLE_ARCHIVE} mxnet_static ${END_WHOLE_ARCHIVE}
      dmlc
      ${mxnet_LINKER_LIBS}
      ${pslite_LINKER_LIBS}
      )
  else()
    target_link_libraries(${PROJECT_NAME}_unit_tests
      ${GTEST_LIBRARY}
      dmlc
      ${nnvm_LINKER_LIBS}
      ${mxnet_LINKER_LIBS}
      mxnet
      ${pslite_LINKER_LIBS}
      )
  endif()

  add_test(AllTestsIn${PROJECT_NAME}UnitTests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}_unit_tests)
else()
  message(WARNING "Google Test not found")
endif()

