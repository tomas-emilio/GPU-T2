#include <gtest/gtest.h>
#include "ConwayGame.h"

//testeamos que cuda produzca los mismos resultados que cpu
TEST(CudaTest, MatchesCPU) {
    //grid de prueba de 3x3
    std::vector<int> initialGrid = {
        0, 1, 0,
        1, 1, 0,
        0, 0, 1
    };


    ConwayGameCuda* cudaGame = createConwayGameCuda(3, 3, false, 16);
    cudaGame->grid = initialGrid;
    
    //creamos instancia cpu para comparar
    ConwayGameSequential cpuGame(3, 3, initialGrid);

    //un paso para ambas
    cudaGame->update();
    cpuGame.update();

    //resultados de ambas
    std::vector<int> cudaResult = cudaGame->getGrid();
    std::vector<int> cpuResult = cpuGame.getGrid();

    //verificar que sean los mismos resultados
    for (size_t i = 0; i < cudaResult.size(); ++i) {
        EXPECT_EQ(cudaResult[i], cpuResult[i]);
    }

    //liberar memoria cuda
    delete cudaGame;
}