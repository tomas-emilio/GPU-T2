set(SRC_FILES
    ConwayGame.cpp
    ConwayGameCuda.cpp
    ConwayGameOpenCL.cpp
    ConwayGameSequential.cpp)

add_library(ConwayGameLib STATIC ${SRC_FILES})
add_compile_definitions(KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ConwayKernel.cl")

target_include_directories(ConwayGameLib PUBLIC ${PROJECT_SOURCE_DIR}/include)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ConwayKernel.cl
               ${CMAKE_CURRENT_BINARY_DIR}/ConwayKernel.cl COPYONLY)

find_package(OpenCL REQUIRED)
target_include_directories(ConwayGameLib PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(ConwayGameLib PRIVATE ${OpenCL_LIBRARIES})