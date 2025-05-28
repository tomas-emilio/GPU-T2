#define N 16
#define BLOCK_SIZE 64
#define RADIUS 3

kernel void conwayStep(global int *inputGrid, global int *outputGrid, int rows, int cols) {
    int worldSize = rows * cols;
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);

    for (int cellId = globalId; cellId < worldSize; cellId += globalSize) {
        
        int x = cellId % worldSize;
        int y = cellId - x;
        int xLeft = (x + cols - 1) % cols;
        int xRight = (x + 1) % cols;
        int yUp = (y + worldSize - cols) % worldSize;
        int yDown = (y + cols) % worldSize;

        int aliveCells = inputGrid[yUp + xLeft] + inputGrid[yUp + x] + inputGrid[yUp + xRight] +
                         inputGrid[y + xLeft] + inputGrid[y + x] + inputGrid[y + xRight] +
                         inputGrid[yDown + xLeft] + inputGrid[yDown + x] + inputGrid[yDown + xRight];

        outputGrid[x + y] = (aliveCells == 3 || (aliveCells == 2 && inputGrid[x + y])) ? 1 : 0;
    }
}