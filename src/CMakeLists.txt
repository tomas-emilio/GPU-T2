set(SRC_FILES
    ConwayGame.cpp
    ConwayGameCuda.cpp
    ConwayGameOpenCL.cpp
    ConwayGameSequential.cpp)

add_library(ConwayGameLib STATIC ${SRC_FILES})
add_compile_definitions(KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernel")

target_include_directories(ConwayGameLib PUBLIC ${PROJECT_SOURCE_DIR}/include)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel/ConwayKernel.cl
               ${CMAKE_CURRENT_BINARY_DIR}/kernel/ConwayKernel.cl COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel/ConwayIfKernel.cl
               ${CMAKE_CURRENT_BINARY_DIR}/kernel/ConwayIfKernel.cl COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernel/ConwayBlockKernel.cl
               ${CMAKE_CURRENT_BINARY_DIR}/kernel/ConwayBlockKernel.cl COPYONLY)

find_package(OpenCL REQUIRED)
target_include_directories(ConwayGameLib PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(ConwayGameLib PRIVATE ${OpenCL_LIBRARIES})