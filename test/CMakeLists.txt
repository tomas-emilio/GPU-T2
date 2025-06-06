if(BUILD_TESTING)
    include(GoogleTest)
    add_executable(ConwayGameTest
        SequentialTest.cpp
        OpenCLTest.cpp
        CudaTest.cpp
    )
    target_link_libraries(ConwayGameTest ConwayGameLib gtest gtest_main)

    gtest_discover_tests(ConwayGameTest)

    add_executable(resultados ../resultados.cpp)
    target_link_libraries(resultados ConwayGameLib)
endif()