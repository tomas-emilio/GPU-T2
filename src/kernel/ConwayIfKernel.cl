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
        int aliveCells = 0;
        for(int dx=-1; dx<=1; dx++){
            for(int dy=-1; dy<=1; dy++){
                if(inputGrid[rows*(y+dy) + (x+dx)]==1){
                    aliveCells++;
                }
            }
        }
        outputGrid[x + y] = (aliveCells == 3 || (aliveCells == 2 && inputGrid[x + y])) ? 1 : 0;
    }
}