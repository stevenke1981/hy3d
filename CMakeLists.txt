cmake_minimum_required(VERSION 3.20)

project(hy3d_cli LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(hy3d_core
    src/backend_paths.cpp
    src/hy3d_backend.cpp
    src/hy3d_cli.cpp
    src/hy3d_gguf.cpp
    src/hy3d_math.cpp
    src/hy3d_model_loader.cpp
    src/hy3d_runtime.cpp
    src/process_runner.cpp
)

target_include_directories(hy3d_core PUBLIC src)

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

add_executable(test_backend_paths tests/test_backend_paths.cpp)
target_link_libraries(test_backend_paths PRIVATE hy3d_core)
add_test(NAME backend_paths COMMAND test_backend_paths)

add_executable(test_process_runner tests/test_process_runner.cpp)
target_link_libraries(test_process_runner PRIVATE hy3d_core)
add_test(NAME process_runner COMMAND test_process_runner)

add_executable(test_result tests/test_result.cpp)
target_link_libraries(test_result PRIVATE hy3d_core)
add_test(NAME result COMMAND test_result)

if (WIN32)
    add_test(
        NAME backend_wrappers
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -File
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_backend_wrappers.ps1
            -RepoRoot ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(
        NAME make_release
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -File
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_make_release.ps1
            -RepoRoot ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(make_release PROPERTIES LABELS "release;slow" TIMEOUT 180)
    add_test(
        NAME download_sources
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass -File
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_download_sources.ps1
            -RepoRoot ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if (Python3_Interpreter_FOUND)
    add_test(
        NAME python_converter
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_convert_hy3d_shape_to_gguf.py
    )
    add_test(
        NAME python_generate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hy3d_generate.py
    )
    add_test(
        NAME python_texture
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hy3d_texture.py
    )
endif()
