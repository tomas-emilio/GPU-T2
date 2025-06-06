#include <gtest/gtest.h>
#include "ConwayGame.h"

TEST(CudaTest, MatchesCPU) {
    std::vector<int> initialGrid = {
        0, 1, 0,
        1, 1, 0,
        0, 0, 1
    };

    ConwayGameCuda* cudaGame = createConwayGameCuda(3, 3, false, 16);
    cudaGame->grid = initialGrid;
    
    ConwayGameSequential cpuGame(3, 3, initialGrid);

    cudaGame->update();
    cpuGame.update();

    std::vector<int> cudaResult = cudaGame->getGrid();
    std::vector<int> cpuResult = cpuGame.getGrid();

    for (size_t i = 0; i < cudaResult.size(); ++i) {
        EXPECT_EQ(cudaResult[i], cpuResult[i]);
    }

    delete cudaGame;
}