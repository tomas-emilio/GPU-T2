#include <vector>
#include <iostream>
#include <CL/opencl.hpp>

class ConwayGame {
protected:
    int rows;
    int cols;
    std::vector<int> grid;
    std::vector<int> nextGrid;

    int getIndex(int row, int col);
    int getValue(int row, int col);
    int countNeighbors(int row, int col);
public:
    // Constructors
    ConwayGame(int rows, int cols);
    ConwayGame(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGame(std::vector<std::vector<int>> matrixGrid);

    // Destructor
    ~ConwayGame();

    // Get the grid
    std::vector<int> getGrid();

    // Randomize the grid
    void randomizeGrid();

    // Print the grid
    void printGrid();

    // Set value in the grid
    void setValue(int row, int col, int value);
};

class ConwayGameSequential : public ConwayGame {
public:
    ConwayGameSequential(int rows, int cols);
    ConwayGameSequential(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGameSequential(std::vector<std::vector<int>> matrixGrid);
    void update();
};

class ConwayGameOpenCL : public ConwayGame {
private:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer deviceGrid;
    cl::Buffer deviceNextGrid;
public:
    ConwayGameOpenCL(int rows, int cols);
    ConwayGameOpenCL(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGameOpenCL(std::vector<std::vector<int>> matrixGrid);

    void initializeOpenCL();
    void update();
};

class ConwayGameCuda : public ConwayGame {
public:
    void update();
};