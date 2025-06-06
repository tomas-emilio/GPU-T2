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

        int aliveCells = 0;
        if(inputGrid[yUp + xLeft]==1){aliveCells++;} 
        if(inputGrid[yUp + x]==1){aliveCells++;} 
        if(inputGrid[yUp + xRight]==1){aliveCells++;}          
        if(inputGrid[y + xLeft]==1){aliveCells++;} 
        if(inputGrid[y + xRight]==1){aliveCells++;}
        if(inputGrid[yDown + xLeft]==1){aliveCells++;} 
        if(inputGrid[yDown + x]==1){aliveCells++;} 
        if(inputGrid[yDown + xRight]){aliveCells++;}

        outputGrid[x + y] = (aliveCells == 3 || (aliveCells == 2 && inputGrid[x + y])) ? 1 : 0;
    }
}