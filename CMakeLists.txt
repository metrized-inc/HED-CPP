cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test-hed)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SOURCE_FILES source/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (
    ${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    "${TORCH_LIBRARIES}")


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif (MSVC)