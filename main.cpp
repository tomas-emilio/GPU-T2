#include "include/ConwayGame.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

int main() {
    std::ofstream csvFile("results.csv");
    csvFile << "GridSize,Sequential(ms),OpenCL(ms),If(ms),BlockSize(ms)\n";
    for (int i=128; i<=4096; i*=2){
        for(int t=0; t<=10; t++){
            int rows = i;
            int cols = i;
            csvFile << i << ",";
            std::cout << "Grid size: " << i << " x " << i << "\n\n";

            // ======= Sequential Game =======
            std::cout << "Sequential Game of Life:\n";
            ConwayGameSequential sequentialGame(rows, cols);
            sequentialGame.randomizeGrid();
            //get the vector to test the same grid
            std::vector<int> vectorGame = sequentialGame.grid;

            auto startSeq = std::chrono::high_resolution_clock::now();
            sequentialGame.update();
            auto endSeq = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> durationSeq = endSeq - startSeq;
            std::cout << "Sequential update took " << durationSeq.count() << " ms\n\n";
            csvFile << durationSeq.count() << ",";

            // ======= OpenCL Game =======
            std::cout << "OpenCL Game of Life:\n";
            ConwayGameOpenCL openclGame(i, i, vectorGame);
            openclGame.initializeOpenCL("ConwayKernel.cl");

            auto startOcl = std::chrono::high_resolution_clock::now();
            openclGame.update();
            auto endOcl = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> durationOcl = endOcl - startOcl;
            std::cout << "OpenCL update took " << durationOcl.count() << " ms\n\n";
            csvFile << durationOcl.count() << ",";

            // ======= OpenCL With If Game =======
            std::cout << "OpenCL Game of Life Block:\n";
            ConwayGameOpenCL openclGameIF(i, i, vectorGame);
            openclGameIF.initializeOpenCL("ConwayBlockKernel.cl");

            auto startOclIF = std::chrono::high_resolution_clock::now();
            openclGameIF.update();
            auto endOclIF = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> durationIfOcl = endOclIF - startOclIF;
            std::cout << "OpenCL IF update took " << durationIfOcl.count() << " ms\n\n";
            csvFile << durationIfOcl.count() << ",";

            // ======= OpenCL With Block Size Game =======
            std::cout << "OpenCL Game of Life If:\n";
            ConwayGameOpenCL openclGameBLOCK(i, i, vectorGame);
            openclGameBLOCK.initializeOpenCL("ConwayIfKernel.cl");

            auto startOclBlock = std::chrono::high_resolution_clock::now();
            openclGameBLOCK.update();
            auto endOclBlock = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> durationOclBlock = endOclBlock - startOclBlock;
            std::cout << "OpenCL BLOCK update took " << durationOclBlock.count() << " ms\n\n";
            csvFile << durationOclBlock.count() << "\n";
        }
    }
    csvFile.close();

    return 0;
}
