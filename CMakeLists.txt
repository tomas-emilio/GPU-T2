cmake_minimum_required(VERSION 3.28)
project(ConwayGame LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

add_subdirectory(src)
add_subdirectory(extern)
add_subdirectory(test)

add_executable(debug_main main.cpp)
target_link_libraries(debug_main ConwayGameLib)
