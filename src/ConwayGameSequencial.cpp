#include "ConwayGame.h"

void ConwayGameSequential::update(){
    for(int i = 0; i < this->rows; ++i) {
        for(int j=0; j < this->cols; ++j) {
            int neighbors = countNeighbors(i, j);
            nextGrid[getIndex(i,j)] = neighbors == 3 || (neighbors == 2 && getValue(i, j)) ? 1 : 0;
        }
    }
    std::swap(grid, nextGrid);
}