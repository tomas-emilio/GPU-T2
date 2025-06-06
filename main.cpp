#include "include/ConwayGame.h"
#include <iostream>
#include <chrono>

int main() {
    const int rows = 512;
    const int cols = 512;

    // ======= Sequential Game =======
    std::cout << "Sequential Game of Life:\n";
    ConwayGameSequential sequentialGame(rows, cols);
    sequentialGame.randomizeGrid();

    auto startSeq = std::chrono::high_resolution_clock::now();
    sequentialGame.update();
    auto endSeq = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> durationSeq = endSeq - startSeq;
    std::cout << "Sequential update took " << durationSeq.count() << " ms\n";

    // ======= OpenCL Game =======
    std::cout << "\nOpenCL Game of Life:\n";
    ConwayGameOpenCL openclGame(rows, cols);
    openclGame.randomizeGrid();
    openclGame.initializeOpenCL();

    auto startOcl = std::chrono::high_resolution_clock::now();
    openclGame.update();
    auto endOcl = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> durationOcl = endOcl - startOcl;
    std::cout << "OpenCL update took " << durationOcl.count() << " ms\n";

    /* ======= Optional CUDA Game Stub =======
    std::cout << "\nCUDA Game of Life (if implemented):\n";
    ConwayGameCuda cudaGame(rows, cols);
    cudaGame.randomizeGrid();

    auto startCuda = std::chrono::high_resolution_clock::now();
    cudaGame.update(); // Only works if implemented
    auto endCuda = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> durationCuda = endCuda - startCuda;
    std::cout << "CUDA update took " << durationCuda.count() << " ms (may be 0 if not implemented)\n";*/

    return 0;
}
