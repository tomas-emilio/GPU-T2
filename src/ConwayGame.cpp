#include <iostream>
#include <vector>
#include "ConwayGame.h"

int ConwayGame::getIndex(int row, int col){
    return row * this->cols + col;
}
int ConwayGame::getValue(int row, int col){
    if (row < 0 || row >= this->rows || col < 0 || col >= this->cols) {
        return 0; // Out of bounds, return dead cell
    }
    return this->grid[getIndex(row, col)];
}
int ConwayGame::countNeighbors(int row, int col){
    return ConwayGame::getValue(row-1, col-1) + ConwayGame::getValue(row-1, col) + ConwayGame::getValue(row-1, col+1) +
           ConwayGame::getValue(row, col-1) + ConwayGame::getValue(row, col+1) +
           ConwayGame::getValue(row+1, col-1) + ConwayGame::getValue(row+1, col) + ConwayGame::getValue(row+1, col+1);
}

// Constructors
ConwayGame::ConwayGame(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    this->grid.resize(rows * cols, 0); // Initialize grid with dead cells
    this->nextGrid.resize(rows * cols, 0); // Initialize nextGrid with dead cells
}
ConwayGame::ConwayGame(int rows, int cols, std::vector<int> vectorGrid){
    this->rows = rows;
    this->cols = cols;
    this->grid = vectorGrid; // Initialize grid with provided vector
    this->nextGrid.resize(rows * cols, 0); // Initialize nextGrid with dead cells
}
ConwayGame::ConwayGame(std::vector<std::vector<int>> matrixGrid){
    this->rows = matrixGrid.size();
    this->cols = matrixGrid[0].size();
    this->grid.resize(rows * cols, 0);
    this->nextGrid.resize(rows * cols, 0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            this->grid[getIndex(i, j)] = matrixGrid[i][j];
        }
    }
}

// Destructor
ConwayGame::~ConwayGame(){}

// Get the grid
std::vector<int> ConwayGame::getGrid(){
    return this->grid;
}

// Randomize the grid
void ConwayGame::randomizeGrid(){
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->grid[getIndex(i, j)] = rand() % 2; // Randomly set cell to alive or dead
        }
    }
}

// Print the grid
void ConwayGame::printGrid(){
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            std::cout << this->grid[getIndex(i, j)] << " ";
        }
        std::cout << std::endl;
    }
}

// Set value in the grid
void ConwayGame::setValue(int row, int col, int value){
    if (row >= 0 && row < this->rows && col >= 0 && col < this->cols) {
        this->grid[getIndex(row, col)] = value;
    }
}