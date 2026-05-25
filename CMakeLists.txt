cmake_minimum_required(VERSION 3.20)

project(hy3d_cli LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hy3d_core
    src/hy3d_backend.cpp
    src/hy3d_cli.cpp
    src/hy3d_gguf.cpp
    src/hy3d_math.cpp
    src/hy3d_model_loader.cpp
    src/hy3d_runtime.cpp
    src/process_runner.cpp
)

target_include_directories(hy3d_core PUBLIC src)

if (MSVC)
    target_compile_options(hy3d_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(hy3d_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(hy3d src/main.cpp)
target_link_libraries(hy3d PRIVATE hy3d_core)

enable_testing()

add_executable(test_cli tests/test_cli.cpp)
target_link_libraries(test_cli PRIVATE hy3d_core)
add_test(NAME cli COMMAND test_cli)

add_executable(test_gguf tests/test_gguf.cpp)
target_link_libraries(test_gguf PRIVATE hy3d_core)
add_test(NAME gguf COMMAND test_gguf)

add_executable(test_gguf_loader tests/test_gguf_loader.cpp)
target_link_libraries(test_gguf_loader PRIVATE hy3d_core)
add_test(NAME gguf_loader COMMAND test_gguf_loader)

add_executable(test_hy3d_runtime tests/test_hy3d_runtime.cpp)
target_link_libraries(test_hy3d_runtime PRIVATE hy3d_core)
add_test(NAME hy3d_runtime COMMAND test_hy3d_runtime)

add_executable(test_hy3d_model_loader tests/test_hy3d_model_loader.cpp)
target_link_libraries(test_hy3d_model_loader PRIVATE hy3d_core)
add_test(NAME hy3d_model_loader COMMAND test_hy3d_model_loader)

add_executable(test_hy3d_math tests/test_hy3d_math.cpp)
target_link_libraries(test_hy3d_math PRIVATE hy3d_core)
add_test(NAME hy3d_math COMMAND test_hy3d_math)

add_executable(test_backend tests/test_backend.cpp)
target_link_libraries(test_backend PRIVATE hy3d_core)
add_test(NAME backend COMMAND test_backend)

if (Python3_Interpreter_FOUND)
    add_test(
        NAME python_converter
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_convert_hy3d_shape_to_gguf.py
    )
endif()
