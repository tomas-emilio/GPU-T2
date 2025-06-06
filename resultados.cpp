#include "include/ConwayGame.h"
#include <iostream>
#include <chrono>
#include <fstream>

//funcion para medir tiempo de ejecucion
double medirTiempo(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    //csv para guardar resultados
    std::ofstream csv("results.csv");
    csv << "Implementation,GridSize,BlockSize,UseIfs,Time Ms\n";
    
    //configuramos experimentos con 
    //tamaños de grids y bloques multiplos y no multiplos de 32
    std::vector<int> gridSizes = {128, 256, 512, 1024};
    std::vector<int> blockSizes = {16, 32, 64, 17, 24}; // múltiplos y no múltiplos de 32
    int iteraciones = 5; //iteraciones a promediar
    
    //ejecutamos experimento para las combinaciones
    for (int gridSize : gridSizes) {
        //creamos un grid aleatorio
        std::vector<int> testGrid(gridSize * gridSize);
        srand(42);
        for (int& cell : testGrid) cell = rand() % 2;
        
        std::cout << "Testeando grid " << gridSize << "x" << gridSize << std::endl;
        
        //secuencial
        ConwayGameSequential seqGame(gridSize, gridSize, testGrid);
        double seqTime = medirTiempo([&]() {
            for (int i = 0; i < iteraciones; ++i) seqGame.update();
        }) / iteraciones;
        csv << "Secuencial," << gridSize << ",0,false," << seqTime << "\n";
        
        //CUDA
        for (int blockSize : blockSizes) {
            //probamos ambos metodos de conteo
            for (bool useIfs : {false, true}) {
                try {
                    ConwayGameCuda* cudaGame = createConwayGameCuda(gridSize, gridSize, useIfs, blockSize);
                    cudaGame->grid = testGrid;
                    
                    //medicion del tiempo de ejecucion
                    double cudaTime = medirTiempo([&]() {
                        for (int i = 0; i < iteraciones; ++i) cudaGame->update();
                    }) / iteraciones;
                    
                    //guardamos en csv
                    csv << "CUDA," << gridSize << "," << blockSize << "," 
                        << (useIfs ? "true" : "false") << "," << cudaTime << "\n";
                    
                    delete cudaGame;
                    std::cout << "  CUDA block=" << blockSize << " ifs=" << useIfs << " time=" << cudaTime << "ms" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "  CUDA falló: " << e.what() << std::endl;
                }
            }
        }
    }
    
    csv.close();
    std::cout << "Resultados guardados en results.csv" << std::endl;
    return 0;
}