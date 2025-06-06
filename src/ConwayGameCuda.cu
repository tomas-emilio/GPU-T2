#include "ConwayGame.h"
#include <cuda_runtime.h>
#include <iostream>

//kernel principal de conway usando bucles para contar vecinos
__global__ void conwayKernel(int* inputGrid, int* outputGrid, int rows, int cols){
    //calcular indice del thread en 2d
    int idx = blockIdx.x * blockDim.x +threadIdx.x;
    int idy = blockIdx.y * blockDim.y +threadIdx.y;

    //verificar limites del grid
    if(idx>=cols || idy >= rows) return;

    int celdaId= idy * cols + idx;
    int vecinosVivos = 0;

    //contar vecinos vivos con bucles anidados
    for(int dy=-1; dy<=1; dy++){
        for(int dx=-1; dx<=1; dx++){
            if(dx==0 && dy==0) continue;
            int ny = idy + dy;
            int nx = idx + dx;
            //verificar que vecino este en los limites
            if(ny>=0 && ny<rows && nx>=0 && nx<cols){
                vecinosVivos += inputGrid[ny*cols +nx];
            }
        }
    }

    //aplicar reglas
    int celdaActual = inputGrid[celdaId];
    outputGrid[celdaId] = (celdaActual == 1) ? 
        ((vecinosVivos == 2 || vecinosVivos == 3) ? 1 : 0) :
        ((vecinosVivos == 3) ? 1 : 0);
}

//kernel alternativo con ifs
__global__ void conwayKernelConIfs(int* inputGrid, int* outputGrid, int rows, int cols){
    //calcular indice en 2d
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    //verificar limites
    if (idx >= cols || idy >= rows) return;
    
    int celdaId = idy * cols + idx;
    int vecinosVivos = 0;

    //vericicar vecinos con el if (tomando como vecinas las 8 celdas que la rodean)
    if (idy > 0 && idx > 0 && inputGrid[(idy-1) * cols + (idx-1)] == 1) vecinosVivos++;
    if (idy > 0 && inputGrid[(idy-1) * cols + idx] == 1) vecinosVivos++;
    if (idy > 0 && idx < cols-1 && inputGrid[(idy-1) * cols + (idx+1)] == 1) vecinosVivos++;
    if (idx > 0 && inputGrid[idy * cols + (idx-1)] == 1) vecinosVivos++;
    if (idx < cols-1 && inputGrid[idy * cols + (idx+1)] == 1) vecinosVivos++;
    if (idy < rows-1 && idx > 0 && inputGrid[(idy+1) * cols + (idx-1)] == 1) vecinosVivos++;
    if (idy < rows-1 && inputGrid[(idy+1) * cols + idx] == 1) vecinosVivos++;
    if (idy < rows-1 && idx < cols-1 && inputGrid[(idy+1) * cols + (idx+1)] == 1) vecinosVivos++;
    
    int celdaActual = inputGrid[celdaId];
    outputGrid[celdaId] = (celdaActual == 1) ? 
        ((vecinosVivos == 2 || vecinosVivos == 3) ? 1 : 0) :
        ((vecinosVivos == 3) ? 1 : 0);
}

//implementacion de la clase cuda
class ConwayGameCudaImpl : public ConwayGameCuda {
private:
    int* d_grid; //ptr a memoria gpu para grid actuall
    int* d_nextGrid; //ptr a gpu para grid siguiente
    size_t gridSize; //tamaño en bytes grid
    bool useIfs; //flag para elegir kernel
    int blockSize; //tamaño de bloque
    
public:
    //constructor
    ConwayGameCudaImpl(int rows, int cols, bool useIfs, int blockSize) 
        : ConwayGameCuda(rows, cols), useIfs(useIfs), blockSize(blockSize) {
        this->rows = rows;
        this->cols = cols;
        this->grid.resize(rows * cols, 0);
        this->nextGrid.resize(rows * cols, 0);
        
        gridSize = rows * cols * sizeof(int);
        cudaMalloc(&d_grid, gridSize);
        cudaMalloc(&d_nextGrid, gridSize);
    }
    
    //para liberar memoria
    ~ConwayGameCudaImpl() {
        cudaFree(d_grid);
        cudaFree(d_nextGrid);
    }
    
    //metodo de actualizacion
    void update() override {
        cudaMemcpy(d_grid, grid.data(), gridSize, cudaMemcpyHostToDevice);
        
        dim3 blockDim(blockSize, blockSize);
        dim3 gridDim((cols + blockSize - 1) / blockSize, (rows + blockSize - 1) / blockSize);
        
        //elegir kernel segun configuracion
        if (useIfs) {
            conwayKernelConIfs<<<gridDim, blockDim>>>(d_grid, d_nextGrid, rows, cols);
        } else {
            conwayKernel<<<gridDim, blockDim>>>(d_grid, d_nextGrid, rows, cols);
        }
        
        //espera ejecucion de todos los hilos
        cudaDeviceSynchronize();
        cudaMemcpy(nextGrid.data(), d_nextGrid, gridSize, cudaMemcpyDeviceToHost);
        std::swap(grid, nextGrid);
    }
};

ConwayGameCuda* createConwayGameCuda(int rows, int cols, bool useIfs, int blockSize) {
    return new ConwayGameCudaImpl(rows, cols, useIfs, blockSize);
}

void ConwayGameCuda::update() {
    throw std::runtime_error("Use factory function");
}