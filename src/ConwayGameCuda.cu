#include "ConwayGame.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void conwayKernel(int* inputGrid, int* outputGrid, int rows, int cols){
    int idx = blockIdx.x * blockDim.x +threadIdx.x;
    int idy = blockIdx.y * blockDim.y +threadIdx.y;

    if(idx>=cols || idy >= rows) return;

    int celdaId= idy * cols + idx;
    int vecinosVivos = 0;

    for(int dy=-1; dy<=1; dy++){
        for(int dx=-1; dx<=1; dx++){
            if(dx==0 && dy==0) continue;
            int ny = idy + dy;
            int nx = idx + dx;
            if(ny>=0 && ny<rows && nx>=0 && nx<cols){
                vecinosVivos += inputGrid[ny*cols +nx];
            }
        }
    }

    int celdaActual = inputGrid[celdaId];
    outputGrid[celdaId] = (celdaActual == 1) ? 
        ((vecinosVivos == 2 || vecinosVivos == 3) ? 1 : 0) :
        ((vecinosVivos == 3) ? 1 : 0);
}

__global__ void conwayKernelConIfs(int* inputGrid, int* outputGrid, int rows, int cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= cols || idy >= rows) return;
    
    int celdaId = idy * cols + idx;
    int vecinosVivos = 0;

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

