
if(NOT MSVC)
  add_custom_target(cpp-package ALL
    DEPENDS mxnet ${CMAKE_SOURCE_DIR}/cpp-package/src/OpWrapperGenerator/OpWrapperGenerator.py
    COMMAND cp $<TARGET_FILE:mxnet> .
    COMMAND python OpWrapperGenerator.py $<TARGET_FILE_NAME:mxnet>
    COMMAND rm $<TARGET_FILE_NAME:mxnet>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/cpp-package/src/OpWrapperGenerator/
    )
endif()

add_subdirectory(example)
